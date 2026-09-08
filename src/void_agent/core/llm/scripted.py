"""The deterministic seam: tests and offline demos script the model, so
framework behavior — not model behavior — is what gets asserted.

A scripted step is usually a literal `ModelStep`; it may instead be a
function of the transcript so far, for the steps that must echo something
the run produced — an id from a tool result, the answer a card came back with.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from void_agent.core.errors import Internal
from void_agent.core.events import EventSender, TextDelta, TextEnd, TextStart
from void_agent.core.llm.interface import (
    ModelStep,
    ToolCall,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
)

ScriptedStep = ModelStep | Callable[[Sequence[TranscriptEntry]], ModelStep]


def say(text: str) -> ModelStep:
    """A scripted step that answers in text."""
    return ModelStep(text=text)


def call(name: str, args: dict[str, Any], *, call_id: str | None = None) -> ModelStep:
    """A scripted step that makes one tool call."""
    return ModelStep(text="", tool_calls=(tool_call(name, args, call_id=call_id),))


def tool_call(name: str, args: dict[str, Any], *, call_id: str | None = None) -> ToolCall:
    """One call for a scripted multi-call step."""
    return ToolCall(call_id=call_id or f"call_{uuid.uuid4().hex[:8]}", name=name, args=args)


def last_tool_return(transcript: Sequence[TranscriptEntry]) -> Any:
    """The newest tool return in the transcript, parsed as JSON — what a
    reactive scripted step reads to echo an id the run minted."""
    for entry in reversed(transcript):
        if isinstance(entry, ToolReturns):
            return json.loads(entry.returns[-1].content)
    raise ValueError("the transcript holds no tool return yet")


class ScriptedLlm:
    """The deterministic seam tests and offline demos script: each `step`
    pops the next scripted step — a literal `ModelStep`, or a function of
    the transcript that builds one — streaming its text through the event
    channel exactly like a live provider would."""

    def __init__(self, steps: Sequence[ScriptedStep]) -> None:
        self._steps = list(steps)

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep:
        if not self._steps:
            raise Internal("scripted llm", RuntimeError("script exhausted before the run ended"))
        scripted = self._steps.pop(0)
        step = scripted if isinstance(scripted, ModelStep) else scripted(transcript)
        if step.text:
            block_id = f"txt_{uuid.uuid4().hex}"
            await events.send(TextStart(id=block_id))
            await events.send(TextDelta(id=block_id, delta=step.text))
            await events.send(TextEnd(id=block_id))
        return step
