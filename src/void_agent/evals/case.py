"""One case: its persona's servers mounted, then each turn run the way a
server runs one — the agent built afresh from the model and the mounted
tools, handed the conversation so far, its stream drained — and written
down whole.

Between turns the conversation grows as a server's would: the person's
message, then the assistant's parts as the next turn's model reads them
(`context_content`, with the suite's cap on a tool result and its choice
of the tool calls that are read back). A question the model puts to the
person is answered from the case's `replies`, in place; with none left,
it ends the turn, and the next turn is the answer, as a person's next
message would be. A turn that fails or runs out of time ends the case.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from void_agent import (
    Agent,
    Answer,
    Attendant,
    EventSender,
    Llm,
    Message,
    PartsAccumulator,
    Role,
    ScriptedHuman,
    Tool,
    Usage,
    context_content,
)
from void_agent.evals.attach import file_part
from void_agent.evals.mount import MountError, Opener, mounted, open_server
from void_agent.evals.person import Person
from void_agent.evals.record import (
    CallLog,
    EventLog,
    Record,
    RecordingLlm,
    failure,
    jsonable,
    usage_record,
)
from void_agent.evals.suite import (
    PERSON_ROLE,
    Case,
    Context,
    Limits,
    Persona,
    SuiteError,
    Turn,
)

Builder = Callable[[Llm, Sequence[Tool]], Agent]
EVENT_BUFFER = 64
# The conversation itself, which every later turn reads: what was said,
# attached, asked and answered, and how a turn ended.
SPOKEN = frozenset({"text", "file", "data-ask", "data-answer", "data-cancelled", "data-error"})


@dataclass(frozen=True, slots=True)
class Setup:
    """What every case of a run shares."""

    build: Builder
    llm: Llm
    open: Opener = open_server
    env: Mapping[str, str] | None = None
    context: Context = field(default_factory=Context)
    limits: Limits = field(default_factory=Limits)
    # The model that plays the person of a case with a brief; None: nobody does.
    person: Llm | None = None
    person_rounds: int = 3
    person_role: str = PERSON_ROLE


async def run_case(case: Case, persona: Persona, setup: Setup, *, key: str = "") -> Record:
    """`case`, asked as `persona`, written down. Never raises for what the
    case did: a server that would not mount, an agent that would not build
    and a turn that failed are each the record's to say."""
    started = time.monotonic()

    def clock() -> float:
        return round(time.monotonic() - started, 3)

    record: Record = {
        "key": key or case.id,
        "id": case.id,
        "persona": persona.name,
        "case": _case(case),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "outcome": "error",
        "turns": [],
    }
    try:
        async with mounted(persona, setup.env or {}, setup.open) as mount:
            record.update(
                catalog=mount.catalog, offered=list(mount.offered), withheld=list(mount.withheld)
            )
            await _converse(case, setup, mount.tools, record, clock)
    except (MountError, SuiteError) as error:
        record["error"] = {"stage": "mount", **failure(error)}
    record["seconds"] = clock()
    record["usage"] = _usage(record["turns"])
    return record


async def _converse(
    case: Case,
    setup: Setup,
    tools: Sequence[tuple[str, Tool]],
    record: Record,
    clock: Callable[[], float],
) -> None:
    """The case's own turns, in order; after each, while the case has a
    person and the agent asked them something, the person's reply as a turn
    of its own."""
    person = (
        Person(setup.person, case.brief, setup.person_rounds, clock, setup.person_role)
        if case.brief and setup.person is not None
        else None
    )
    human: Attendant | None = person
    if person is None and case.replies:
        human = ScriptedHuman(case.replies)
    chat = _Chat(setup, tools, record, clock, human, person)
    try:
        for turn in case.turns:
            if not await chat.take(turn, by="case"):
                return
            while person is not None and person.can_reply():
                reply = await person.reply_after_turn()
                if reply is None:
                    break
                if not await chat.take(Turn(reply), by="person"):
                    return
    finally:
        if person is not None:
            record["person"] = person.record()


class _Chat:
    """One case's conversation as the agent is handed it, turn by turn."""

    def __init__(
        self,
        setup: Setup,
        tools: Sequence[tuple[str, Tool]],
        record: Record,
        clock: Callable[[], float],
        human: Attendant | None,
        person: Person | None,
    ) -> None:
        self._setup = setup
        self._tools = tools
        self._record = record
        self._clock = clock
        self._human = human
        self._person = person
        self._history: list[Message] = []
        self._offered = frozenset(tool.name for _, tool in tools)

    async def take(self, turn: Turn, *, by: str) -> bool:
        """Run one turn; False when the case cannot go on after it."""
        setup, record, clock = self._setup, self._record, self._clock
        try:
            user_parts = _user_parts(turn)
        except SuiteError as error:
            record.update(outcome="error", error={"stage": "attach", **failure(error)})
            return False
        self._history.append(Message(Role.USER, context_content(user_parts)))
        if self._person is not None:
            self._person.said(turn.text)
        calls = CallLog(clock)
        llm = RecordingLlm(setup.llm, clock)
        try:
            agent = setup.build(
                llm, [calls.recorded(tool, source) for source, tool in self._tools]
            )
        except Exception as error:
            record.update(outcome="error", error={"stage": "build", **failure(error)})
            return False
        events = EventLog(clock)
        done = await _run(agent, list(self._history), self._human, events, setup.limits, clock)
        parts = done.parts
        record["turns"].append(
            {
                "user": {
                    "text": turn.text,
                    "attach": [path.name for path in turn.attach],
                    "by": by,
                },
                "outcome": done.outcome,
                "answer": done.answer,
                "ask": done.ask,
                "asks": events.asks(),
                "error": done.error,
                "at": done.at,
                "seconds": done.seconds,
                "steps": llm.steps,
                "calls": events.calls(llm.steps, calls),
                "parts": parts,
                "events": events.events,
            }
        )
        # The tools as the model saw them: the same on every turn.
        if llm.specs is not None:
            record.setdefault("specs", llm.specs)
        record["outcome"] = done.outcome
        if self._person is not None:
            self._person.heard(_words(done))
        if done.outcome in ("failed", "cancelled"):
            return False
        reread = context_content(
            _reread(parts, setup.context, self._offered),
            tool_output_limit=setup.context.tool_output_limit,
        )
        if reread:
            self._history.append(Message(Role.ASSISTANT, (_joined(reread),)))
        return True


def _words(done: _Done) -> str:
    """What the agent said to the person this turn, as they read it."""
    if isinstance(done.answer, str):
        return done.answer
    if done.ask is not None:
        return str(done.ask.get("question", ""))
    if done.error is not None:
        return str(done.error.get("text", ""))
    return "" if done.answer is None else str(done.answer)


@dataclass(frozen=True, slots=True)
class _Done:
    outcome: str
    parts: list[dict[str, Any]]
    at: float
    seconds: float
    answer: Any = None
    ask: Record | None = None
    error: Record | None = None


async def _run(
    agent: Agent,
    history: list[Message],
    human: Attendant | None,
    log: EventLog,
    limits: Limits,
    clock: Callable[[], float],
) -> _Done:
    """One turn, its stream drained while it runs, the way a server folds
    it into parts — with the markers a server writes at the end: how long
    it took, then a cancellation or a readable failure."""
    sender, queue = EventSender.channel(EVENT_BUFFER)
    accumulator = PartsAccumulator()
    at, started = clock(), time.monotonic()

    async def drain() -> None:
        while True:
            event = await queue.get()
            log.add(event)
            accumulator.apply(event)

    drainer = asyncio.create_task(drain())
    run = asyncio.create_task(agent.run(history, sender, human=human))
    outcome, answer, ask, error = "failed", None, None, None
    try:
        async with asyncio.timeout(limits.turn_seconds):
            result = await run
        if isinstance(result, Answer):
            outcome, answer = "answered", jsonable(result.value)
        else:
            outcome, ask = "asked", jsonable(asdict(result))
    except TimeoutError:
        outcome = "cancelled"
        error = {"type": "Timeout", "text": f"the turn took longer than {limits.turn_seconds:g} s"}
    except Exception as raised:
        error = failure(raised)
    finally:
        drainer.cancel()
    while not queue.empty():
        event = queue.get_nowait()
        log.add(event)
        accumulator.apply(event)
    seconds = round(time.monotonic() - started, 3)
    parts = accumulator.into_parts()
    parts.append({"type": "data-elapsed", "data": {"seconds": round(seconds, 1)}})
    if outcome == "cancelled":
        parts.append({"type": "data-cancelled", "data": {}})
    elif outcome == "failed" and error is not None:
        parts.append({"type": "data-error", "data": {"text": error["text"]}})
    return _Done(outcome, parts, at=at, seconds=seconds, answer=answer, ask=ask, error=error)


def _user_parts(turn: Turn) -> list[dict[str, Any]]:
    """The person's message as a client sends it: the text, then each
    attachment as a `file` part."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": turn.text}]
    for path in turn.attach:
        part = file_part(path)
        if isinstance(part, str):
            raise SuiteError(f"cannot attach {path}: {part}")
        parts.append(part)
    return parts


def _reread(
    parts: list[dict[str, Any]], context: Context, offered: frozenset[str]
) -> list[dict[str, Any]]:
    """The parts of a turn the next turn reads: all of them, or — when the
    suite names the tools whose calls are read back — the conversation,
    those calls, and the plan as it last stood."""
    if context.tools == "all":
        return parts
    tools = offered if context.tools == "mounted" else frozenset(context.tools)
    last_plan = max(
        (index for index, part in enumerate(parts) if part.get("type") == "data-plan"),
        default=-1,
    )
    return [
        part
        for index, part in enumerate(parts)
        if part.get("type") in SPOKEN
        or (part.get("type") == "dynamic-tool" and part.get("toolName") in tools)
        or index == last_plan
    ]


def _joined(content: Sequence[Any]) -> Any:
    """Assistant history is one text: an assistant turn carries no
    attachment, so its content is one text part already."""
    (part,) = content
    return part


def _usage(turns: Sequence[Record]) -> Record:
    total = Usage(input=0, output=0)
    trips = 0
    for turn in turns:
        for step in turn["steps"]:
            usage = step.get("usage")
            if usage is not None:
                total = total + Usage(**usage)
                trips += 1
    return {**(usage_record(total) or {}), "round_trips": trips}


def _case(case: Case) -> Record:
    return {
        "turns": [
            {"text": turn.text, "attach": [str(path) for path in turn.attach]}
            for turn in case.turns
        ],
        "replies": list(case.replies),
        "tags": list(case.tags),
        "intent": case.intent,
        "gap": case.gap,
        "expect": jsonable(asdict(case.expect)),
    }
