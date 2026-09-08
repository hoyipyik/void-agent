"""Reflection is a projection of the model's self-assessment.

`reflect` is an ordinary built-in tool, registered like any other — not every
agent needs it, but any agent may have it. Its value is structural: the
model gets a sanctioned pause to assess its own progress, and the
assessment becomes durable — a `ReflectionMade` event for the UI, a
`data-reflection` part in storage, a compact line in the next turn's
context via `context_text`. Lessons survive the turn the way the plan does.

The runtime judges nothing here (a wrong reflection is the model's problem,
same as a wrong plan); the discipline lives in the tool description.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from void_agent.core.events import EventSender, ReflectionMade
from void_agent.core.tool import Tool

REFLECT_DESCRIPTION = (
    "Pause and assess your own progress. Use it after a setback (a failed or "
    "surprising tool result), before an irreversible action, and before submitting "
    "a long task's answer. State FACTS — what actually happened, with evidence, not "
    "what you hoped; give a verdict; when the verdict is not on_track, name the "
    "problems and the concrete adjustment you will make next. Your reflections are "
    "kept in the conversation, so they also brief the next turn."
)


class ReflectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: str
    verdict: Literal["on_track", "adjust", "blocked"]
    problems: list[str] = []
    adjustment: str | None = None


def reflect_tool() -> Tool:
    """The built-in `reflect` tool. Stateless: the reflection's home is the
    transcript (the call's own arguments), the event stream, and the parts."""

    async def reflect(input: ReflectInput, events: EventSender) -> str:
        await events.send(
            ReflectionMade(
                verdict=input.verdict,
                facts=input.facts,
                problems=tuple(input.problems),
                adjustment=input.adjustment,
            )
        )
        return f"reflection recorded ({input.verdict})"

    return Tool(name="reflect", description=REFLECT_DESCRIPTION, handler=reflect)
