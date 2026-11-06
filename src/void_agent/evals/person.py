"""The person a case is asked by, played by a model.

A case that says who is asking and what they want (`brief`) is answered by
a person, not a script: when the agent asks them something — by calling
`ask_user`, or in the words of its reply — a model reads the brief and the
chat so far and answers as they would, in their language and tone. When
the agent asked them nothing, they say nothing, and the case goes on with
its own next turn or ends.

The person sees what a person in the chat would: their brief, what they
said, and what the agent said to them. Never a tool call or a result, and
never the agent's prompt. Each of their moves is kept with the record:
what they were asked, what they said, what it cost.

They answer an approval card by not answering it: a gated call is
signed by whoever owns the decision, and a brief does not.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from void_agent import Agent, Answer, Ask, Llm, Message, RunError, Usage
from void_agent.evals.record import Record, RecordingLlm, failure, usage_record
from void_agent.evals.suite import PERSON_ROLE

PERSON = """\
You play {role}. You are not an assistant: you are the one asking.
What you want, and what you know, is in your brief; the assistant cannot see it.

Write as that person writes in a work chat: in the language and tone of your
first message, short, casual, often without punctuation; small typos are fine.
Say only what your brief says you know or want. When the assistant asks for
something your brief does not cover, say you do not know, or answer as such a
person plausibly would, but never invent a code, a figure or a name. Do not
repeat the brief word for word, and never say that you are playing a part.
"""

AFTER_TURN = """\
Read the assistant's last message. When it needs an answer from you to go on
(it asked you a question, or asked you to choose or to confirm), give that
answer. When it answered you, or only offered something you did not ask for,
or asked you nothing, say nothing: hand in reply null."""

ASKED = "The assistant asks you this now, and waits for your answer: give it."


class Move(BaseModel):
    reply: str | None = Field(
        description="What you write next, as the person; null to say nothing"
    )


def _rendered(shown: str) -> list[Message]:
    return [Message.user(shown)]


class Person:
    """One case's person: their brief, the chat as they saw it, and every
    move they made. Also the run's attendant, so a question the agent puts
    with `ask_user` reaches them in place."""

    def __init__(
        self,
        llm: Llm,
        brief: str,
        rounds: int,
        clock: Callable[[], float],
        role: str = PERSON_ROLE,
    ) -> None:
        self.moves: list[Record] = []
        self._llm = llm
        self._brief = brief
        self._role = role
        self._rounds = rounds
        self._clock = clock
        self._chat: list[tuple[str, str]] = []

    @property
    def replies(self) -> int:
        return sum(1 for move in self.moves if move.get("reply") is not None)

    def can_reply(self) -> bool:
        return self.replies < self._rounds

    def said(self, text: str) -> None:
        self._chat.append(("You", text))

    def heard(self, text: str) -> None:
        self._chat.append(("Assistant", text))

    async def reply_after_turn(self) -> str | None:
        """Their next message, when the agent's last one asked them for it."""
        return await self._move(AFTER_TURN, question=None)

    async def answer(self, ask: Ask) -> str | None:
        """The agent's `ask_user`, answered in place; None ends the wait."""
        if not self.can_reply():
            return None
        lines = [ASKED, "", ask.question]
        if ask.options:
            lines.append("Options: " + " | ".join(ask.options))
        reply = await self._move("\n".join(lines), question=ask.question)
        if reply is not None:
            self.said(reply)
        return reply

    async def approve(self, ask: Ask) -> bool | None:
        return None

    def record(self) -> Record:
        return {"brief": self._brief, "moves": self.moves}

    async def _move(self, ending: str, *, question: str | None) -> str | None:
        move: Record = {"at": self._clock(), "question": question}
        self.moves.append(move)
        llm = RecordingLlm(self._llm, self._clock)
        agent = (
            Agent(llm, "person", "the person asking the assistant")
            .with_system(PERSON.format(role=self._role))
            .prompt(_rendered)
            .output(Move, instructions="Hand in what you write next, or null.")
        )
        started = time.monotonic()
        try:
            result: Any = await agent.run(self._shown(ending))
        except RunError as error:
            move.update(reply=None, error=failure(error))
            result = None
        move["seconds"] = round(time.monotonic() - started, 3)
        total = Usage(input=0, output=0)
        for step in llm.steps:
            if step.get("usage"):
                total = total + Usage(**step["usage"])
        move["usage"] = usage_record(total)
        if isinstance(result, Answer) and isinstance(result.value, Move):
            reply = (result.value.reply or "").strip() or None
            move["reply"] = reply
            return reply
        move.setdefault("reply", None)
        return None

    def _shown(self, ending: str) -> str:
        chat = "\n".join(f"{speaker}: {text}" for speaker, text in self._chat)
        return (
            f"Your brief, which only you can see:\n{self._brief}\n\n"
            f"The chat so far:\n{chat}\n\n{ending}"
        )
