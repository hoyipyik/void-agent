"""Who answers for the person while a run is alive, and the two ready-made
ways to be that person.

An `Attendant` attends a run: `Agent.run(..., human=…)` makes it ambient
for the whole tree (`attended`), so a question from any depth — the
model's `ask_user`, a tool's approval gate — reaches the same person, and
its answer returns to the loop that asked, in place. No handle travels
through user code; the runtime consults the attendant itself.

Two methods, by the question's source. A gate asks for a SIGNATURE:
`approve` returns a boolean, and the gate never sees what button or word
produced it — that translation belongs to the application's edge. The
model's `ask_user` asks in words: `answer` returns the person's words,
whatever the ask's `kind`. Either returns None when the wait was dropped.

Two attendants ship, so an application implements nothing: `ScriptedHuman`
is the deterministic seam tests and offline examples run on, and
`HumanChannel` is the one a real application attends through — its
questions arrive on a queue as `Question`s, each carrying the slot its
reply goes into. How a question reaches whoever attends, and what happens
when nobody does, is `channel`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

from void_agent.core.ask import Ask


class Attendant(Protocol):
    """Whoever answers for the person while a run is alive — the channel.

    `approve` is asked for a gate's signature (the ask carries `call`) and
    returns the decision as a boolean; `answer` is asked for the model's
    `ask_user` and returns the person's words. Either returns None when
    the wait was dropped: nobody answered within the attendant's patience,
    or the person is gone. The runtime treats None as the end of the run."""

    async def approve(self, ask: Ask) -> bool | None: ...

    async def answer(self, ask: Ask) -> str | None: ...


_attendant: ContextVar[Attendant | None] = ContextVar("void_agent.attendant", default=None)


@contextmanager
def attended(human: Attendant | None) -> Generator[None]:
    """Make `human` the attendant for the block. `Agent.run` wraps the run
    in it, so every nested run, tool, and gate in the tree consults the
    same person — no handle travels through user code."""
    token = _attendant.set(human)
    try:
        yield
    finally:
        _attendant.reset(token)


def attendant() -> Attendant | None:
    return _attendant.get()


class ScriptedHuman:
    """The deterministic seam for the person: each question pops the next
    scripted answer — a bool for a signature, a str for words; an exhausted
    script, or a scripted None, drops the wait. A scripted value of the
    wrong shape is a test bug and says so. Every question asked is kept in
    `asked`."""

    def __init__(self, answers: Sequence[bool | str | None] = ()) -> None:
        self._answers = list(answers)
        self.asked: list[Ask] = []

    async def approve(self, ask: Ask) -> bool | None:
        self.asked.append(ask)
        scripted = self._answers.pop(0) if self._answers else None
        if scripted is not None and not isinstance(scripted, bool):
            raise TypeError(f"a signature is scripted as a bool, not {scripted!r}")
        return scripted

    async def answer(self, ask: Ask) -> str | None:
        self.asked.append(ask)
        scripted = self._answers.pop(0) if self._answers else None
        if scripted is not None and not isinstance(scripted, str):
            raise TypeError(f"words are scripted as a str, not {scripted!r}")
        return scripted


class Question:
    """An `Ask` in flight with its reply slot. `signature` says which shape
    the reply takes: a bool for a gate's card, words for the model's.

    Which button or word means "yes" is the application's translation to
    make before `reply`: a signature is replied to with a bool, the model's
    question with a str, and the wrong shape is refused as a programming
    error."""

    __slots__ = ("_reply", "ask")

    def __init__(self, ask: Ask, reply: asyncio.Future[bool | str | None]) -> None:
        self.ask = ask
        self._reply = reply

    @property
    def signature(self) -> bool:
        return self.ask.call is not None

    def reply(self, value: bool | str) -> bool:
        """Answer in place; the frame that asked continues. Returns False
        when the wait is already over: answered, dropped, or the run gone."""
        if self.signature and not isinstance(value, bool):
            raise TypeError(f"a signature is replied to with a bool, not {value!r}")
        if not self.signature and not isinstance(value, str):
            raise TypeError(f"words are replied to with a str, not {value!r}")
        return self._settle(value)

    def drop(self) -> bool:
        """Give up on the question: the run ends with the card open."""
        return self._settle(None)

    def _settle(self, value: bool | str | None) -> bool:
        if self._reply.done():
            return False
        self._reply.set_result(value)
        return True


class HumanChannel:
    """The ready-made `Attendant`: each question goes on the queue, and the
    frame that asked waits on its reply for as long as `patience` allows
    (None: indefinitely). The queue is bounded, so the reader must drain
    it while the run is alive, as with an event channel.

    `HumanChannel.channel(capacity)` is the person as a pair, the way
    `EventSender.channel` is the stream: an `Attendant` to pass as
    `run(..., human=…)`, and a queue the application reads. Every question
    the run tree asks — the model's `ask_user`, a gate three layers down —
    arrives on the queue as a `Question`: the `Ask`, plus the slot its
    reply goes into. `reply` wakes the frame that asked, in place. Nobody
    matches answers to questions; each question carries its own way back.

        human, questions = HumanChannel.channel(8)
        run = asyncio.create_task(agent.run(input, events, human=human))
        question = await questions.get()
        question.reply(True)            # a signature; words for `ask_user`

    No reply within `patience` drops the wait — the run ends with the card
    open, exactly as when nobody attends — and so does `drop`, right away.
    A reply after either is refused and returns False."""

    def __init__(self, queue: asyncio.Queue[Question], patience: float | None) -> None:
        self._queue = queue
        self._patience = patience

    @classmethod
    def channel(
        cls, capacity: int, *, patience: float | None = None
    ) -> tuple[HumanChannel, asyncio.Queue[Question]]:
        """The attendant to pass as `human=` and the queue of its questions."""
        if capacity <= 0:
            raise ValueError("human channel capacity must be greater than zero")
        queue: asyncio.Queue[Question] = asyncio.Queue(capacity)
        return cls(queue, patience), queue

    async def approve(self, ask: Ask) -> bool | None:
        decision = await self._put_and_wait(ask)
        return decision if isinstance(decision, bool) else None

    async def answer(self, ask: Ask) -> str | None:
        words = await self._put_and_wait(ask)
        return words if isinstance(words, str) else None

    async def _put_and_wait(self, ask: Ask) -> bool | str | None:
        reply: asyncio.Future[bool | str | None] = asyncio.get_running_loop().create_future()
        try:
            async with asyncio.timeout(self._patience):
                await self._queue.put(Question(ask, reply))
                return await reply
        except TimeoutError:
            # The expiry cancelled `reply`, so a late `reply()` is refused.
            return None
