"""One turn, run on the app's own loop.

The runtime runs in-process: `agent.run(history, events, human=…)` is a
task, and `drain` reads its stream and its questions concurrently — the
loop a streaming server would run — folding every
event into `parts` with the `PartsAccumulator` and handing each one to
`on_event` with the array as it stands, so a view can render it live.
Every question the run tree asks — the model's `ask_user`, a gate three
layers down — arrives as a `Question` and goes to `on_question`; its
reply wakes the frame that asked, in place. Nothing here is Textual.

The parts come back with the transport markers a server would add: a
stop is a `data-cancelled` part, a readable failure a `data-error` one
(`public_text` — Internal detail reaches neither the log nor the model).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from void_agent import (
    Agent,
    AgentEvent,
    EventSender,
    HumanChannel,
    Message,
    PartsAccumulator,
    Question,
    RunError,
    public_text,
)

EVENT_BUFFER = 64
ASK_BUFFER = 8

Parts = list[dict[str, Any]]
OnEvent = Callable[[AgentEvent, Parts], Awaitable[None]]
OnQuestion = Callable[[Question], None]


class Turn:
    """The run as a task, started right here, and the way to drain it.
    Built inside the loop; `cancel` is Escape."""

    def __init__(self, agent: Agent, history: list[Message]) -> None:
        events, self._queue = EventSender.channel(EVENT_BUFFER)
        human, self._questions = HumanChannel.channel(ASK_BUFFER, patience=None)
        self._accumulator = PartsAccumulator()
        self._run: asyncio.Task[Any] = asyncio.create_task(agent.run(history, events, human=human))

    @property
    def running(self) -> bool:
        return not self._run.done()

    def cancel(self) -> None:
        """Stop the run: it ends marked cancelled, nothing else changes."""
        if not self._run.done():
            self._run.cancel()

    async def drain(self, on_event: OnEvent, on_question: OnQuestion) -> Parts:
        """Read the stream and the questions until the run ends, and return
        the folded parts, transport markers included."""
        run, queue, questions = self._run, self._queue, self._questions
        accumulator = self._accumulator
        pending_get: asyncio.Task[AgentEvent] | None = None
        pending_question: asyncio.Task[Question] | None = None
        try:
            while True:
                if pending_get is None:
                    pending_get = asyncio.ensure_future(queue.get())
                if pending_question is None:
                    pending_question = asyncio.ensure_future(questions.get())
                done, _ = await asyncio.wait(
                    {pending_get, pending_question, run}, return_when=asyncio.FIRST_COMPLETED
                )
                if pending_question in done:
                    on_question(pending_question.result())
                    pending_question = None
                if pending_get in done:
                    event = pending_get.result()
                    pending_get = None
                    accumulator.apply(event)
                    await on_event(event, accumulator.into_parts())
                elif run in done:
                    break
        finally:
            if pending_get is not None:
                pending_get.cancel()
            if pending_question is not None:
                pending_question.cancel()
            if not run.done():
                run.cancel()
        while not queue.empty():
            event = queue.get_nowait()
            accumulator.apply(event)
            await on_event(event, accumulator.into_parts())
        parts = accumulator.into_parts()
        try:
            await run
        except asyncio.CancelledError:
            if not run.cancelled():
                raise  # the caller itself is being cancelled: let it
            parts.append({"type": "data-cancelled", "data": {}})
        except RunError as error:
            # Readable failures reach the log and the next turn's model;
            # Internal detail stays out of both (`public_text`).
            parts.append({"type": "data-error", "data": {"text": public_text(error)}})
        return parts
