"""The loop: one turn driven to its ending.

Deliberately dumb: each step the model decides to hand in, ask the
person, or go again; the loop relays, counts, and caps — the model is the
one that adapts, so a new model generation never means a new loop.

Preemption comes before side effects. A valid `final_answer` ends the
step before any sibling call runs and before the transcript grows. An
`ask_user` goes to the person before any sibling side effect: answered,
the answer is that call's result and the step goes on; unanswered, the
turn ends right there and nothing else in the step acts. A question a
tool call raised and nobody answered (`Unanswered`) unwinds out of the
call and ends the turn the same way — with the card open on the session,
where the next message wakes the model.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from void_agent.core.agent.dispatch import (
    ToolRegistration,
    dispatch,
    error_return,
    tool_output_text,
    tool_return,
)
from void_agent.core.agent.outcome import Answer, TurnResult
from void_agent.core.agent.rules import (
    ASK_USER,
    ASK_USER_DESCRIPTION,
    FINAL_ANSWER,
    FINAL_ANSWER_DESCRIPTION,
    MUST_SPEAK_RULE,
    MUST_SUBMIT_EMPTY_RULE,
    MUST_SUBMIT_RULE,
)
from void_agent.core.ask import Ask, ask_adapter
from void_agent.core.errors import Exhausted, Internal, RunError
from void_agent.core.events import EventSender, StepStart
from void_agent.core.human import Unanswered, ask_words
from void_agent.core.llm import (
    AssistantStep,
    Llm,
    ModelStep,
    ToolCall,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    UserText,
)


class Stall(enum.Enum):
    """Why a step made no progress. The latest stall names the failure once
    the stall budget is spent."""

    PLAIN_TEXT_ANSWER = enum.auto()
    EMPTY_ANSWER = enum.auto()
    NO_TOOL_CALL_ACCEPTED = enum.auto()


@dataclass(frozen=True, slots=True)
class Finished:
    answer: Any


@dataclass(frozen=True, slots=True)
class LeftOpen:
    """The step put a question to the person and nobody answered it: the
    turn ends here with the card open on the session. A question that was
    answered never reaches this outcome — the answer is the call's result
    and the step goes on."""

    ask: Ask


@dataclass(frozen=True, slots=True)
class Progressed:
    pass


@dataclass(frozen=True, slots=True)
class Stalled:
    stall: Stall


StepOutcome = Finished | LeftOpen | Progressed | Stalled


@dataclass(frozen=True, slots=True)
class Loop:
    """An agent's loop, configured once; `run` drives one turn over the
    transcript it is given, growing it in place. The loop holds no run
    state of its own, so one serves any number of concurrent turns."""

    name: str
    llm: Llm
    tools: Mapping[str, ToolRegistration]
    output_adapter: TypeAdapter[Any] | None
    output_instructions: str
    asks_user: bool
    max_steps: int
    max_retries: int

    def _specs(self) -> list[ToolSpec]:
        """Describe tools and loop controls from the same settings execution uses.
        `final_answer` uses the tool-call format, but submits the agent's result;
        it is never registered or dispatched as an ordinary tool."""
        specs = [
            ToolSpec(
                name=registration.tool.name,
                description=registration.tool.description,
                input_schema=registration.tool.input_schema,
            )
            for registration in self.tools.values()
        ]
        if self.output_adapter is not None:
            specs.append(
                ToolSpec(
                    name=FINAL_ANSWER,
                    description=f"{FINAL_ANSWER_DESCRIPTION}\n\n{self.output_instructions}".strip(),
                    input_schema=self.output_adapter.json_schema(),
                )
            )
        if self.asks_user:
            specs.append(
                ToolSpec(
                    name=ASK_USER,
                    description=ASK_USER_DESCRIPTION,
                    input_schema=ask_adapter.json_schema(),
                )
            )
        return specs

    async def run(self, transcript: list[TranscriptEntry], events: EventSender) -> TurnResult:
        specs = self._specs()
        consecutive_stalls = 0
        for step_index in range(self.max_steps):
            await events.send(StepStart(step=step_index + 1))
            try:
                step = await self.llm.step(transcript, specs, events)
            except (RunError, asyncio.CancelledError):
                raise
            except Exception as error:
                raise Internal("call model", error) from error

            match await self._apply(step, transcript, events):
                case Finished(answer):
                    return Answer(answer)
                case LeftOpen(ask):
                    return ask
                case Progressed():
                    consecutive_stalls = 0
                case Stalled(stall):
                    consecutive_stalls += 1
                    if consecutive_stalls > self.max_retries:
                        raise self._stall_error(stall, consecutive_stalls)

        raise Exhausted(f"agent {self.name} hit its step limit before finishing")

    async def _apply(
        self, step: ModelStep, transcript: list[TranscriptEntry], events: EventSender
    ) -> StepOutcome:
        """Apply one model step to the transcript: record what the model
        said, accept a submission if one validates, put its questions to
        the person, and run the remaining tool calls concurrently — the
        sequencing rule in the system prompt makes same-step calls
        independent of each other."""
        if not step.tool_calls:
            return self._apply_text_only(step, transcript)

        # A valid submission ends the run before any side effect runs and
        # before the transcript grows: the model said "this is the answer",
        # so nothing else in the step should still act.
        submission_errors: dict[int, ToolReturn] = {}
        for index, tool_call in enumerate(step.tool_calls):
            if self.output_adapter is None or tool_call.name != FINAL_ANSWER:
                continue
            try:
                return Finished(self.output_adapter.validate_python(tool_call.args))
            except ValidationError as error:
                submission_errors[index] = error_return(
                    tool_call, f"invalid {FINAL_ANSWER}: {error}"
                )

        transcript.append(AssistantStep(step))

        returns: list[ToolReturn] = []
        actions: list[ToolCall] = []
        answered = False
        for index, tool_call in enumerate(step.tool_calls):
            if index in submission_errors:
                # Reuse the validation failure, preserving the model's call order.
                # A valid submission would already have finished the step above.
                returns.append(submission_errors[index])
            elif self.asks_user and tool_call.name == ASK_USER:
                # The question goes to the person now, before any sibling
                # side effect: answered, it is this call's result and the
                # step goes on; unanswered, the turn ends here and nothing
                # else in the step acts.
                try:
                    ask = ask_adapter.validate_python(tool_call.args).to_ask()
                except ValidationError as error:
                    returns.append(error_return(tool_call, f"invalid {ASK_USER}: {error}"))
                    continue
                try:
                    answer = await ask_words(ask, events)
                except Unanswered as unanswered:
                    return LeftOpen(unanswered.ask)
                returns.append(tool_return(tool_call, tool_output_text({"answer": answer})))
                answered = True
            else:
                actions.append(tool_call)

        outcomes = await asyncio.gather(
            *(dispatch(call, self.tools.get(call.name), events) for call in actions),
            return_exceptions=True,
        )

        # Cancellation outranks every other failure: the Stop button must
        # never be swallowed because a sibling call happened to crash first.
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome

        progressed = answered
        failure: BaseException | None = None
        unanswered: Unanswered | None = None
        for outcome in outcomes:
            if isinstance(outcome, Unanswered):
                unanswered = unanswered if unanswered is not None else outcome
            elif isinstance(outcome, BaseException):
                failure = failure if failure is not None else outcome
            else:
                progressed = progressed or outcome.progressed
                returns.append(outcome.part)
        if failure is not None:
            raise failure
        if unanswered is not None:
            # A question nobody answered unwound out of a call: this run
            # ends the way the run that asked did. The card is on the
            # stream, and the session keeps it.
            return LeftOpen(unanswered.ask)

        if returns:
            transcript.append(ToolReturns(returns=tuple(returns)))
        return Progressed() if progressed else Stalled(Stall.NO_TOOL_CALL_ACCEPTED)

    def _apply_text_only(self, step: ModelStep, transcript: list[TranscriptEntry]) -> StepOutcome:
        """A step with no tool calls: a text agent is finished — the text IS
        the answer; a typed agent is nudged toward `final_answer` and
        stalls. An EMPTY text is never an answer for either: the model is
        nudged to actually speak."""
        if self.output_adapter is None and step.text.strip():
            return Finished(step.text)
        if step.text or step.raw is not None:
            transcript.append(AssistantStep(step))
        if self.output_adapter is None:
            transcript.append(UserText(MUST_SPEAK_RULE))
            return Stalled(Stall.EMPTY_ANSWER)
        if not step.text.strip():
            # An empty step from a typed agent gets a truthful nudge — it
            # did not "answer in plain text"; it produced nothing at all.
            transcript.append(UserText(MUST_SUBMIT_EMPTY_RULE))
            return Stalled(Stall.EMPTY_ANSWER)
        transcript.append(UserText(MUST_SUBMIT_RULE))
        return Stalled(Stall.PLAIN_TEXT_ANSWER)

    def _stall_error(self, stall: Stall, consecutive_stalls: int) -> Exhausted:
        """The run-ending error once the stall budget is spent, named after
        the way the last step stalled."""
        match stall:
            case Stall.PLAIN_TEXT_ANSWER:
                return Exhausted(
                    f"agent {self.name} kept answering in text instead of calling final_answer"
                )
            case Stall.EMPTY_ANSWER:
                return Exhausted(f"agent {self.name} kept finishing without any message text")
            case Stall.NO_TOOL_CALL_ACCEPTED:
                return Exhausted(
                    f"agent {self.name} made no progress"
                    f" for {consecutive_stalls} consecutive steps"
                )
