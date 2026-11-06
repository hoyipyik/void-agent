"""Writing a turn down while it runs, from the three places that see it.

The model, wrapped: each round-trip is kept with what the model read that
it had not read before — the system prompt, the history and the question
on the first step, then the tool results exactly as they went back — and
what it returned: its text, its calls, its reasoning where the provider
hands that over, its account. The tools, wrapped: each call's input, its
whole result, how long it took, and the detail of a failure that neither
the model nor the stream is shown. The stream: every event in order, with
the time it arrived.

Nothing here changes what the agent does. A wrapped model returns the
step it was given and a wrapped tool the result, and the wrappers
re-raise what they catch.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic_core import to_jsonable_python

from void_agent import (
    HELD_TOOL_OUTPUT,
    AgentEvent,
    AssistantStep,
    AssistantText,
    EventSender,
    Exhausted,
    ImageContent,
    Internal,
    Llm,
    ModelStep,
    PdfContent,
    Rejected,
    RunError,
    SystemText,
    TextContent,
    TextDelta,
    Tool,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    Unanswered,
    Usage,
    UserContent,
    UserText,
    public_text,
)

Clock = Callable[[], float]
Record = dict[str, Any]
# What a provider hands back that no person can read: replayed to the
# provider verbatim, written down only as its length.
OPAQUE = frozenset({"encrypted_content", "signature", "data"})


def usage_record(usage: Usage | None) -> Record | None:
    return None if usage is None else asdict(usage)


def failure(error: BaseException) -> Record:
    """A failure as the record keeps it: its type, the words a person would
    be shown, and the whole traceback, which only the record keeps."""
    text = public_text(error) if isinstance(error, RunError) else str(error)
    return {
        "type": type(error).__name__,
        "text": text,
        "detail": "".join(traceback.format_exception(error)).strip(),
    }


def jsonable(value: Any) -> Any:
    try:
        return to_jsonable_python(value)
    except Exception:
        return repr(value)


def chars(value: Any) -> int:
    """The length of a result as the model reads it: a string as it is,
    anything else as compact JSON."""
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(jsonable(value), separators=(",", ":"), ensure_ascii=False))


# ── the model ─────────────────────────────────────────────────────────────


class RecordingLlm:
    """The model, with each round-trip written down. One per turn."""

    def __init__(self, inner: Llm, clock: Clock) -> None:
        self.steps: list[Record] = []
        self.specs: list[Record] | None = None
        self._inner = inner
        self._clock = clock
        self._seen = 0
        # Which recorded step returned each ModelStep, by identity: the
        # transcript hands this run's own steps back on the next one.
        self._numbers: dict[int, int] = {}

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep:
        if self.specs is None:
            self.specs = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
                for spec in tools
            ]
        record: Record = {
            "step": len(self.steps) + 1,
            "at": self._clock(),
            "read": [self._entry(entry) for entry in transcript[self._seen :]],
        }
        self._seen = len(transcript)
        self.steps.append(record)
        started = time.monotonic()
        try:
            step = await self._inner.step(transcript, tools, events)
        except BaseException as error:
            record["seconds"] = round(time.monotonic() - started, 3)
            record["error"] = failure(error)
            raise
        record["seconds"] = round(time.monotonic() - started, 3)
        record["text"] = step.text
        record["calls"] = [
            {"id": call.call_id, "name": call.name, "args": call.args} for call in step.tool_calls
        ]
        record["usage"] = usage_record(step.usage)
        reasoning = _reasoning(step.raw)
        if reasoning:
            record["reasoning"] = reasoning
        if step.raw is not None:
            record["raw"] = _scrubbed(jsonable(step.raw))
        self._numbers[id(step)] = record["step"]
        return step

    def _entry(self, entry: TranscriptEntry) -> Record:
        match entry:
            case SystemText(text):
                return {"system": text}
            case UserContent(content):
                return {"user": [_content(part) for part in content]}
            case UserText(text):
                return {"user_text": text}
            case AssistantText(text):
                return {"assistant": text}
            case AssistantStep(step):
                number = self._numbers.get(id(step))
                if number is not None:
                    return {"assistant_step": number}
                return {"assistant": step.text, "calls": [call.name for call in step.tool_calls]}
            case ToolReturns(returns):
                return {
                    "tool_returns": [
                        {"id": item.call_id, "name": item.name, "content": item.content}
                        for item in returns
                    ]
                }


def _content(part: object) -> Record:
    if isinstance(part, TextContent):
        return {"text": part.text}
    if isinstance(part, ImageContent):
        return {"image": part.media_type, "bytes": len(part.data)}
    if isinstance(part, PdfContent):
        return {"pdf": part.filename, "bytes": len(part.data)}
    return {"unknown": type(part).__name__}


def _reasoning(raw: Any) -> list[str]:
    """What a provider says of its reasoning, where it says anything in
    words: OpenAI's summaries, Anthropic's thinking."""
    if not isinstance(raw, list):
        return []
    found: list[str] = []
    for item in cast("list[Any]", jsonable(raw)):
        if not isinstance(item, dict):
            continue
        block = cast("dict[str, Any]", item)
        if block.get("type") == "reasoning":
            for summary in cast("list[Any]", block.get("summary") or []):
                text = (
                    cast("dict[str, Any]", summary).get("text")
                    if isinstance(summary, dict)
                    else None
                )
                if text:
                    found.append(str(text))
        elif block.get("type") == "thinking" and block.get("thinking"):
            found.append(str(block["thinking"]))
    return found


def _scrubbed(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (f"<{len(str(item))} chars>" if key in OPAQUE and item else _scrubbed(item))
            for key, item in cast("dict[str, Any]", value).items()
        }
    if isinstance(value, list):
        return [_scrubbed(item) for item in cast("list[Any]", value)]
    return value


# ── the tools ─────────────────────────────────────────────────────────────


class CallLog:
    """Every invocation of the wrapped tools in one turn, in the order they
    started."""

    def __init__(self, clock: Clock) -> None:
        self.invocations: list[Record] = []
        self._clock = clock

    def recorded(self, tool: Tool, source: str) -> Tool:
        """`tool`, with each call written down. Its own gate, validation
        and handler run as they would; the wrapper re-raises what they
        raise."""

        async def handler(input: dict[str, Any], events: EventSender) -> Any:
            entry: Record = {
                "name": tool.name,
                "source": source,
                "input": input,
                "at": self._clock(),
            }
            self.invocations.append(entry)
            started = time.monotonic()
            try:
                output = await tool.invoke(input, events)
            except (Rejected, Exhausted) as error:
                entry.update(status="refused", error=str(error))
                raise
            except Internal as error:
                entry.update(
                    status="broken", error=public_text(error), detail=failure(error)["detail"]
                )
                raise
            except Unanswered:
                entry.update(status="held")
                raise
            except asyncio.CancelledError:
                entry.update(status="cancelled")
                raise
            finally:
                entry["seconds"] = round(time.monotonic() - started, 3)
            entry.update(status="ok", output=output, chars=chars(output))
            return output

        return Tool(
            name=tool.name,
            description=tool.description,
            handler=handler,
            input_schema=tool.input_schema,
        )

    def take(self, name: str, input: Any) -> Record | None:
        """The first invocation not yet taken that matches a call the
        stream reported: same tool, same input."""
        key = _key(input)
        for entry in self.invocations:
            if not entry.get("taken") and entry["name"] == name and _key(entry["input"]) == key:
                entry["taken"] = True
                return entry
        return None


def _key(input: Any) -> str:
    return json.dumps(jsonable(input), sort_keys=True, ensure_ascii=False)


# ── the stream ────────────────────────────────────────────────────────────


class EventLog:
    """Every event of a turn, in order, each with the time it arrived. A
    run of text deltas of one block is kept as one."""

    def __init__(self, clock: Clock) -> None:
        self.events: list[Record] = []
        self._clock = clock

    def add(self, event: AgentEvent) -> None:
        last = self.events[-1] if self.events else None
        if (
            isinstance(event, TextDelta)
            and last is not None
            and last["type"] == "TextDelta"
            and last["id"] == event.id
        ):
            last["delta"] += event.delta
            return
        fields = asdict(event) if is_dataclass(event) else {}
        self.events.append({"type": type(event).__name__, "at": self._clock(), **jsonable(fields)})

    def asks(self) -> list[Record]:
        """Each question put to the person while the turn ran, with the
        answer it got in place, or None when none came."""
        asked: dict[str, Record] = {}
        for event in self.events:
            if event["type"] == "AskIssued":
                asked[event["ask_id"]] = {"question": event["question"], "answer": None}
            elif event["type"] == "AskAnswered" and event["ask_id"] in asked:
                asked[event["ask_id"]]["answer"] = event["value"]
        return list(asked.values())

    def calls(self, steps: Sequence[Record], log: CallLog) -> list[Record]:
        """Each call the stream reported, with what the tool's own record
        adds: which step made it, how long it ran, the detail of a failure."""
        step_of = {call["id"]: step["step"] for step in steps for call in step.get("calls", [])}
        made: dict[str, Record] = {}
        for event in self.events:
            kind, id = event["type"], str(event.get("tool_call_id"))
            if kind == "ToolInputAvailable":
                made[id] = {
                    "id": id,
                    "name": event["tool_name"],
                    "step": step_of.get(id),
                    "input": event["input"],
                    "at": event["at"],
                }
            elif kind in ("ToolOutputAvailable", "ToolOutputError") and id in made:
                entry = made[id]
                entry["seconds"] = round(event["at"] - entry["at"], 3)
                if kind == "ToolOutputError":
                    entry.update(status="error", error=event["error_text"])
                elif event["output"] == HELD_TOOL_OUTPUT:
                    entry.update(status="held", output=event["output"])
                else:
                    entry.update(status="ok", output=event["output"], chars=chars(event["output"]))
        for entry in made.values():
            invocation = log.take(entry["name"], entry["input"])
            if invocation is not None:
                entry.update(
                    {
                        key: value
                        for key, value in invocation.items()
                        if key not in ("taken", "name", "input")
                    }
                )
            entry.setdefault("status", "unfinished")
        return list(made.values())
