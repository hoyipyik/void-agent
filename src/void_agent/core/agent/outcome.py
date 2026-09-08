"""How a turn ends: `Answer`, or an `Ask` nobody answered (errors travel
as exceptions).

Fewer endings mean fewer branches for the application; every intermediate
state is folded into the shape of a message — a question nobody answered
while the run was alive is an ask card at the tail of the session, with no
live process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from void_agent.core.ask import Ask


@dataclass(frozen=True, slots=True)
class Answer:
    """The turn finished: the answer text, or the validated output instance
    for a typed agent."""

    value: Any


TurnResult = Answer | Ask
