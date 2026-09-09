"""The exact wire JSON for each event: the Vercel AI SDK UI message
stream, with the typed data events riding the protocol's `data-*`
extension point. Transport code only serializes what `to_wire` returns."""

from __future__ import annotations

from typing import Any

from void_agent.core.events.types import (
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
)


def _ask_data(event: AskIssued) -> dict[str, Any]:
    data: dict[str, Any] = {
        "askId": event.ask_id,
        "kind": event.kind,
        "question": event.question,
        "options": list(event.options) if event.options is not None else None,
        "payload": event.payload,
    }
    if event.call is not None:
        data["call"] = event.call.to_data()
    return data


def to_wire(event: AgentEvent) -> dict[str, Any]:
    """The exact Vercel UI message stream JSON for one event.

    Every tool the runtime executes is "dynamic" in protocol terms — none are
    client-side, statically registered — so tool events always carry
    `"dynamic": true`. The typed data events serialize as `data-*` parts.
    """
    match event:
        case Start(message_id):
            return {"type": "start", "messageId": message_id}
        case Finish():
            return {"type": "finish"}
        case Error(error_text):
            return {"type": "error", "errorText": error_text}
        case StepStart(step):
            return {"type": "data-step", "data": {"step": step}}
        case TextStart(id=block_id):
            return {"type": "text-start", "id": block_id}
        case TextDelta(id=block_id, delta=delta):
            return {"type": "text-delta", "id": block_id, "delta": delta}
        case TextEnd(id=block_id):
            return {"type": "text-end", "id": block_id}
        case ToolInputStart(tool_call_id, tool_name):
            return {
                "type": "tool-input-start",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "dynamic": True,
            }
        case ToolInputAvailable(tool_call_id, tool_name, input=tool_input):
            return {
                "type": "tool-input-available",
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": tool_input,
                "dynamic": True,
            }
        case ToolOutputAvailable(tool_call_id, output):
            return {
                "type": "tool-output-available",
                "toolCallId": tool_call_id,
                "output": output,
                "dynamic": True,
            }
        case ToolOutputError(tool_call_id, error_text):
            return {
                "type": "tool-output-error",
                "toolCallId": tool_call_id,
                "errorText": error_text,
                "dynamic": True,
            }
        case UsageReported(usage):
            return {
                "type": "data-usage",
                "data": {
                    "input": usage.input,
                    "output": usage.output,
                    "cacheRead": usage.cache_read,
                    "cacheWrite": usage.cache_write,
                },
            }
        case PlanUpdated(items):
            return {"type": "data-plan", "data": {"items": items}}
        case ReflectionMade() as reflection:
            return {
                "type": "data-reflection",
                "data": {
                    "verdict": reflection.verdict,
                    "facts": reflection.facts,
                    "problems": list(reflection.problems),
                    "adjustment": reflection.adjustment,
                },
            }
        case AskIssued() as ask:
            return {"type": "data-ask", "data": _ask_data(ask)}
        case AskAnswered(ask_id, value):
            return {"type": "data-answer", "data": {"askId": ask_id, "value": value}}
        case AskDropped(ask_id):
            return {"type": "data-ask-dropped", "data": {"askId": ask_id}}
        case Progress(kind, data):
            return {"type": f"data-{kind}", "data": data}
