"""Folding the wire stream into a UIMessage `parts` array — the persisted
half of the protocol. The caller owns when and where to store; this module
only knows how events fold. Framing and step rhythm are never persisted."""

from __future__ import annotations

from typing import Any, cast

from void_agent.core.events import (
    AgentEvent,
    AskAnswered,
    AskDropped,
    AskIssued,
    Error,
    Finish,
    PlanUpdated,
    Progress,
    ReflectionMade,
    Start,
    StepStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
    ToolOutputError,
    UsageReported,
    to_wire,
)

_TERMINAL_TOOL_STATES = ("output-available", "output-error")


def _new_tool_part(tool_call_id: str, tool_name: str) -> dict[str, Any]:
    """A tool part as it first appears: input still streaming. The Vercel
    part schema requires `toolName` in every state, so the defensive path
    (an update with no preceding input-start) uses a placeholder name until
    the real one arrives."""
    return {
        "type": "dynamic-tool",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "state": "input-streaming",
    }


class PartsAccumulator:
    """Folds wire events into a UIMessage `parts` array for persistence."""

    def __init__(self) -> None:
        self._parts: list[dict[str, Any]] = []
        # Each text part's index by its stream id. The closing event is a
        # no-op, so entries simply live as long as the accumulator.
        self._block_index: dict[str, int] = {}

    def apply(self, event: AgentEvent) -> None:
        match event:
            case TextStart(id=block_id):
                self._open_block(block_id, {"type": "text", "text": ""})
            case TextDelta(id=block_id, delta=delta):
                index = self._block_index.get(block_id)
                if index is not None:
                    self._parts[index]["text"] += delta
            case TextEnd():
                pass
            case ToolInputStart(tool_call_id, tool_name):
                self._parts.append(_new_tool_part(tool_call_id, tool_name))
            case ToolInputAvailable(tool_call_id, tool_name, input=tool_input):
                part = self._open_tool_part(tool_call_id)
                part["toolName"] = tool_name
                part["input"] = tool_input
                part["state"] = "input-available"
            case ToolOutputAvailable(tool_call_id, output):
                part = self._open_tool_part(tool_call_id)
                part["output"] = output
                part["state"] = "output-available"
            case ToolOutputError(tool_call_id, error_text):
                part = self._open_tool_part(tool_call_id)
                part["errorText"] = error_text
                part["state"] = "output-error"
            # A data event's wire frame IS its part: `{"type": "data-<kind>",
            # "data": …}` on the stream and in storage alike.
            case (
                UsageReported()
                | PlanUpdated()
                | ReflectionMade()
                | AskIssued()
                | AskAnswered()
                | Progress()
            ):
                self._parts.append(to_wire(event))
            case AskDropped(ask_id):
                # Not a part of its own: the card it refers to is marked, so
                # a replay renders it dropped and the model reads it so.
                ask = self._open_ask_part(ask_id)
                if ask is not None:
                    ask["dropped"] = True
            # Protocol framing and step rhythm are not content; nothing to
            # persist — a replayed conversation carries results, not pacing.
            case Start() | Finish() | Error() | StepStart():
                pass

    def _open_block(self, block_id: str, part: dict[str, Any]) -> None:
        self._parts.append(part)
        self._block_index[block_id] = len(self._parts) - 1

    def _open_tool_part(self, tool_call_id: str) -> dict[str, Any]:
        """The most recent still-open tool part for this call id, created
        defensively if an update arrives with no preceding input-start.

        Finished parts never match: call ids are only unique per provider
        completion, so a nested call can share its parent's id, and the
        parent's output must not overwrite the finished child part."""
        for part in reversed(self._parts):
            if (
                part.get("type") == "dynamic-tool"
                and part.get("toolCallId") == tool_call_id
                and part.get("state") not in _TERMINAL_TOOL_STATES
            ):
                return part
        part = _new_tool_part(tool_call_id, "unknown")
        self._parts.append(part)
        return part

    def _open_ask_part(self, ask_id: str) -> dict[str, Any] | None:
        """The `data` of the most recent ask card with this id."""
        for part in reversed(self._parts):
            if part.get("type") != "data-ask":
                continue
            data = part.get("data")
            if isinstance(data, dict) and data.get("askId") == ask_id:  # pyright: ignore[reportUnknownMemberType]
                return cast("dict[str, Any]", data)
        return None

    def into_parts(self) -> list[dict[str, Any]]:
        return self._parts
