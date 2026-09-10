"""How a question travels: from wherever it was raised, to the person, and
back to the same frame.

`ask_signature` (a gate's question) and `ask_words` (the model's) send the
card out on the stream as `AskIssued`, put it to whoever attends the run,
and hand the decision back to the caller — the tool call or loop that
asked keeps going right there, with `AskAnswered` on the stream.

Nobody attending, or a dropped wait (a timeout, the person left), is
`Unanswered`, a BaseException like `CancelledError`: it unwinds every
layer between the question and the root — workflow handlers, sub-agent
boundaries — without any of them having to relay it, and `except
Exception` cannot swallow it. The run that asked ends with the `Ask`,
every run above it ends the same way, and the session keeps the card
(`AskDropped` marks it): the next message wakes the model, which reads
what happened and carries on. A layer that can genuinely go on without
the answer catches it by name.
"""

from __future__ import annotations

from void_agent.core.ask import Ask
from void_agent.core.events import AskAnswered, AskDropped, AskIssued, EventSender
from void_agent.core.human.attendant import Attendant, attendant


class Unanswered(BaseException):
    """Nobody answered the question while the run was alive — nobody was
    attending, or the attendant gave up waiting. Every run it passes
    through ends with the same `Ask`; the session keeps the card."""

    ask: Ask

    def __init__(self, ask: Ask) -> None:
        super().__init__(ask.question)
        self.ask = ask


async def ask_signature(ask: Ask, events: EventSender) -> bool:
    """A gate's question to whoever attends the run: True to run the call,
    False to refuse it. Raises `Unanswered` when no decision came."""
    human = await _issue(ask, events)
    return await _resolve(ask, events, await human.approve(ask))


async def ask_words(ask: Ask, events: EventSender) -> str:
    """The model's question to whoever attends the run: the person's words,
    which become the `ask_user` call's result. Raises `Unanswered` when no
    answer came."""
    human = await _issue(ask, events)
    return await _resolve(ask, events, await human.answer(ask))


async def _issue(ask: Ask, events: EventSender) -> Attendant:
    """The card goes out on the stream; with nobody attending it simply
    stays open on the session, and the run that asked ends."""
    await events.send(
        AskIssued(
            ask_id=ask.ask_id,
            kind=ask.kind,
            question=ask.question,
            options=ask.options,
            payload=ask.payload,
            call=ask.call,
        )
    )
    human = attendant()
    if human is None:
        raise Unanswered(ask)
    return human


async def _resolve[Decision: (bool, str)](
    ask: Ask, events: EventSender, decision: Decision | None
) -> Decision:
    """The answer comes back as `AskAnswered`, or `AskDropped` when the
    attendant gave up waiting."""
    if decision is None:
        await events.send(AskDropped(ask_id=ask.ask_id))
        raise Unanswered(ask)
    await events.send(AskAnswered(ask_id=ask.ask_id, value=decision))
    return decision
