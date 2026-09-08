"""One registered tool call through its lifecycle: its event mode, the
input events, the invocation, and the outcome classified for the model —
an output, a hold, a readable refusal, or a private failure. What the
model reads back is a contract: the `ERROR: ` prefix never drifts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from void_agent.core.agent.rules import HELD_TOOL_OUTPUT, INTERNAL_TOOL_ERROR
from void_agent.core.errors import Exhausted, Internal, Rejected
from void_agent.core.events import (
    EventMode,
    EventSender,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
    ToolOutputError,
)
from void_agent.core.human import Unanswered
from void_agent.core.llm import ToolCall, ToolReturn
from void_agent.core.tool import Tool


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """A tool and its event visibility in one agent; no execution logic.
    The same tool can be registered elsewhere with a different event mode."""

    tool: Tool
    event_mode: EventMode


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """What one executed tool call produced: the transcript part, and whether
    the call counted as progress (a refusal does not)."""

    progressed: bool
    part: ToolReturn


def tool_return(call: ToolCall, content: str) -> ToolReturn:
    return ToolReturn(call_id=call.call_id, name=call.name, content=content)


def error_return(call: ToolCall, message: str) -> ToolReturn:
    """The model-facing encoding of a refused call. The `ERROR: ` prefix is a
    contract the model reacts to; every rejection must go through here so the
    prefix cannot drift between paths."""
    return tool_return(call, f"ERROR: {message}")


def tool_output_text(output: Any) -> str:
    """A structured tool result as transcript text, without adding JSON
    quotes around an already-plain string."""
    if isinstance(output, str):
        return output
    return json.dumps(output, separators=(",", ":"), ensure_ascii=False)


async def dispatch(
    call: ToolCall, registration: ToolRegistration | None, parent: EventSender
) -> CallOutcome:
    """Run one registered tool (None: the model named an unknown tool).
    `Unanswered` and `Internal` propagate; everything
    else comes back as a `CallOutcome`."""
    events = parent.with_mode(registration.event_mode) if registration is not None else parent

    if events.is_live():
        await events.send(ToolInputStart(tool_call_id=call.call_id, tool_name=call.name))
        await events.send(
            ToolInputAvailable(tool_call_id=call.call_id, tool_name=call.name, input=call.args)
        )

    try:
        if registration is None:
            raise Rejected(f"unknown tool: {call.name}")
        output = await registration.tool.invoke(call.args, events)
    except Unanswered:
        # Nothing ran: the person never answered the question this call
        # raised. The card is already on the stream; the tool part closes
        # on that fact, and the question keeps unwinding — the model meets
        # both on the session next turn.
        if events.is_live():
            await events.send(
                ToolOutputAvailable(tool_call_id=call.call_id, output=HELD_TOOL_OUTPUT)
            )
        raise
    except (Rejected, Exhausted) as error:
        # A sub-agent that exhausts its own budget reads as a rejection at
        # this boundary: either way, the calling model may route around it.
        if events.is_live():
            await events.send(ToolOutputError(tool_call_id=call.call_id, error_text=str(error)))
        return CallOutcome(progressed=False, part=error_return(call, str(error)))
    except Internal:
        # An internal failure stays private — neither the stream nor the
        # model sees its detail.
        if events.is_live():
            await events.send(
                ToolOutputError(tool_call_id=call.call_id, error_text=INTERNAL_TOOL_ERROR)
            )
        raise

    if events.is_live():
        await events.send(ToolOutputAvailable(tool_call_id=call.call_id, output=output))
    return CallOutcome(progressed=True, part=tool_return(call, tool_output_text(output)))
