"""The one producer handle the whole run tree shares. A bare sender is a
free no-op; a bounded channel applies backpressure, so the receiver must
drain concurrently with the run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from void_agent.core.events.types import AgentEvent, Progress
from void_agent.core.events.visibility import EventMode, admit


@dataclass(frozen=True, slots=True)
class EventSender:
    """A small, immutable producer handle shared by the whole run tree.

    A bare `EventSender()` has no queue, so sending becomes a no-op. A bounded
    queue applies backpressure: the receiver must be drained concurrently with
    the run so event production cannot stall unobserved.
    """

    _queue: asyncio.Queue[AgentEvent] | None = None
    _mode: EventMode = EventMode.ALL

    @classmethod
    def channel(cls, capacity: int) -> tuple[EventSender, asyncio.Queue[AgentEvent]]:
        """Create a fully visible, bounded event channel."""
        if capacity <= 0:
            raise ValueError("event channel capacity must be greater than zero")
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue(capacity)
        return cls(queue, EventMode.ALL), queue

    def is_live(self) -> bool:
        """Whether an event built for this sender can be observed at all.

        Emitters on hot paths use it to skip constructing events nobody can
        receive; correctness never depends on it, since `send` filters."""
        return self._queue is not None and self._mode is not EventMode.HIDDEN

    async def send(self, event: AgentEvent) -> None:
        admitted = admit(self._mode, event)
        if admitted is not None and self._queue is not None:
            await self._queue.put(admitted)

    async def progress(self, kind: str, data: Any) -> None:
        """Emit a `data-<kind>` progress event."""
        await self.send(Progress(kind=kind, data=data))

    def with_mode(self, mode: EventMode) -> EventSender:
        """Choose the events visible from a nested function, workflow, or
        agent. A child can never widen an ancestor's mode because the
        strictest mode on the path wins."""
        return EventSender(self._queue, max(self._mode, mode))
