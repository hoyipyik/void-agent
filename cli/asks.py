"""The desk: the questions a turn is waiting on, by ask id, and the one
the composer is answering in words.

The application's edge — where an answer's shape is settled, never in
core. A signature card replies with a
boolean; the model's question with words: an input question opens the
composer as it arrives, a choice question only when its last row, Other,
is chosen. One question takes words at a time; a reply to a question the
desk no longer holds, or in the wrong shape, is dropped — the wait is
over, or the card sent what it could not have."""

from __future__ import annotations

from void_agent import Question


class Desk:
    def __init__(self) -> None:
        self._open: dict[str, Question] = {}
        # The question the composer is answering, if one takes words.
        self.answering: Question | None = None

    def arrive(self, question: Question) -> bool:
        """Keep the question. True when it takes words and the composer
        should open for it now — an input question with nobody ahead of
        it. Every other card holds the keys itself."""
        self._open[question.ask.ask_id] = question
        if question.ask.kind != "input" or question.signature:
            return False
        return self._claim(question)

    def words(self, ask_id: str) -> bool:
        """The person wants to answer this card in words. True when the
        composer should open for it now."""
        question = self._open.get(ask_id)
        if question is None or question.signature:
            return False
        return self._claim(question)

    def _claim(self, question: Question) -> bool:
        if self.answering is not None:
            return False
        self.answering = question
        return True

    def answer(self, ask_id: str, value: bool | str) -> bool:
        """Reply to the question waiting under that card. The card already
        translated its choice into the shape the question takes; a stale
        answer (the wait is over) or one in the wrong shape is dropped."""
        question = self._open.pop(ask_id, None)
        if question is None or question.signature != isinstance(value, bool):
            return False
        question.reply(value)
        if self.answering is question:
            self.answering = None
        return True

    def clear(self) -> None:
        """The turn is over: nothing here can be answered any more."""
        self._open.clear()
        self.answering = None
