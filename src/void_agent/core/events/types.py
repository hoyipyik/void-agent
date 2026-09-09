"""The event vocabulary: one frozen dataclass per thing a run can report.

Events are observation, never state — losing one must not affect
correctness. Reserved data kinds are refused at `Progress` construction so
a tool can never forge a provenance-carrying part."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from void_agent.core.ask import Call
from void_agent.core.usage import Usage


@dataclass(frozen=True, slots=True)
class Start:
    message_id: str


@dataclass(frozen=True, slots=True)
class Finish:
    pass


@dataclass(frozen=True, slots=True)
class Error:
    error_text: str


@dataclass(frozen=True, slots=True)
class StepStart:
    """The loop opened model step `step` (1-based). Pure UI progress: the
    accumulator never persists it, and a replayed conversation shows no
    steps — a projection may always be dropped."""

    step: int


@dataclass(frozen=True, slots=True)
class TextStart:
    id: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    id: str
    delta: str


@dataclass(frozen=True, slots=True)
class TextEnd:
    id: str


@dataclass(frozen=True, slots=True)
class ToolInputStart:
    tool_call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolInputAvailable:
    tool_call_id: str
    tool_name: str
    input: Any


@dataclass(frozen=True, slots=True)
class ToolOutputAvailable:
    tool_call_id: str
    output: Any


@dataclass(frozen=True, slots=True)
class ToolOutputError:
    tool_call_id: str
    error_text: str


@dataclass(frozen=True, slots=True)
class UsageReported:
    """One model round-trip cost this many tokens, as the provider counted
    them. Reported by the loop the moment the step returns — before the
    step's calls run, so a call that crashes never loses the account. It
    carries no step number: a sub-agent's round-trips report through the
    same stream (visibility passes it as progress) and must not disturb
    the root's step rhythm. Persisted as a `data-usage` part, so a session
    can be tallied from what it kept; the model never reads it."""

    usage: Usage


@dataclass(frozen=True, slots=True)
class PlanUpdated:
    """The model rewrote its task list (the built-in `update_plan` tool).
    The whole list replaces the previous one — last write wins, the UI
    renders it as the assistant's own progress narration, never as system
    fact."""

    items: list[Any]


@dataclass(frozen=True, slots=True)
class ReflectionMade:
    """The model paused to assess its own progress (the built-in `reflect`
    tool). A projection like the plan: the UI renders it as the assistant's
    self-assessment, the parts carry it into the next turn's context, and
    losing it never affects correctness."""

    verdict: str
    facts: str
    problems: tuple[str, ...]
    adjustment: str | None


@dataclass(frozen=True, slots=True)
class AskIssued:
    """A question for the human is on the table — the model's `ask_user`
    or a tool's approval gate. The UI renders an answer form or approval
    card. While someone attends the run the answer comes back through the
    attendant and `AskAnswered` follows; otherwise the turn ends with the
    card open, and the answer is the next conversation message. `call` is
    the gated tool call an approval signs — runtime-written, so the card
    renders exactly what will run."""

    ask_id: str
    kind: str
    question: str
    options: tuple[str, ...] | None
    payload: dict[str, Any] | None
    call: Call | None = None


@dataclass(frozen=True, slots=True)
class AskAnswered:
    """The attendant answered while the run was alive; the loop that asked
    continues in place. A gate's signature is a bool, the model's question
    gets words. Folds into a `data-answer` part — the same shape a user's
    answer message carries."""

    ask_id: str
    value: bool | str


@dataclass(frozen=True, slots=True)
class AskDropped:
    """The attendant gave up waiting: the run ends with the card open and
    marked dropped, so the next turn's model reads that no answer came."""

    ask_id: str


RESERVED_DATA_KINDS = frozenset(
    {
        "step",
        "plan",
        "ask",
        "ask-dropped",
        "answer",
        "trigger",
        "cancelled",
        "error",
        "reflection",
        "usage",
        "elapsed",
    }
)


@dataclass(frozen=True, slots=True)
class Progress:
    """A tool-defined progress payload: `{"type": "data-<kind>", "data": ...}`.

    Reserved kinds are refused at construction: a tool must not be able to
    forge the parts that carry provenance — a user's answer, an ask, the
    plan, the step rhythm, a trigger, or a transport marker. Those are
    produced only by the runtime's typed events or appended as parts by the
    application itself."""

    kind: str
    data: Any

    def __post_init__(self) -> None:
        if self.kind in RESERVED_DATA_KINDS:
            raise ValueError(
                f"data-{self.kind} is a reserved part kind and cannot be emitted as tool progress"
            )


AgentEvent = (
    Start
    | Finish
    | Error
    | StepStart
    | TextStart
    | TextDelta
    | TextEnd
    | ToolInputStart
    | ToolInputAvailable
    | ToolOutputAvailable
    | ToolOutputError
    | UsageReported
    | PlanUpdated
    | ReflectionMade
    | AskIssued
    | AskAnswered
    | AskDropped
    | Progress
)
