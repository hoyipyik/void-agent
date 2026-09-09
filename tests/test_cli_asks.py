"""The desk: the questions a turn is waiting on, and which one the
composer is answering in words."""

from __future__ import annotations

import asyncio

from cli.session.asks import Desk

from void_agent import Ask, Call, Question


def question(kind: str = "input", *, call: Call | None = None, ask_id: str = "a1") -> Question:
    ask = Ask(ask_id=ask_id, kind=kind, question="?", options=None, payload=None, call=call)
    return Question(ask, asyncio.get_running_loop().create_future())


def signature(ask_id: str = "s1") -> Question:
    return question("approval", call=Call("create_draft", {"total": 120}), ask_id=ask_id)


async def test_an_input_question_takes_words_and_the_composer_opens() -> None:
    desk = Desk()
    asked = question("input")
    assert desk.arrive(asked)
    assert desk.answering is asked


async def test_a_choice_question_waits_on_the_card_until_other_is_chosen() -> None:
    desk = Desk()
    asked = question("choice")
    assert not desk.arrive(asked)
    assert desk.answering is None
    assert desk.words(asked.ask.ask_id)
    assert desk.answering is asked
    assert not desk.words(asked.ask.ask_id)  # already open for it


async def test_a_signature_is_never_answered_in_words() -> None:
    desk = Desk()
    asked = signature()
    assert not desk.arrive(asked)
    assert not desk.words(asked.ask.ask_id)
    assert desk.answering is None


async def test_one_question_at_a_time_takes_words() -> None:
    desk = Desk()
    first, second = question(ask_id="a1"), question(ask_id="a2")
    assert desk.arrive(first)
    assert not desk.arrive(second)
    assert desk.answering is first
    assert desk.answer("a1", "x")
    assert desk.answering is None


async def test_an_answer_reaches_the_question_and_leaves_the_desk() -> None:
    desk = Desk()
    asked = question()
    desk.arrive(asked)
    assert desk.answer("a1", "blue")
    assert asked._reply.result() == "blue"  # pyright: ignore[reportPrivateUsage]
    assert not desk.answer("a1", "again")  # stale: the wait is over


async def test_the_wrong_shape_is_dropped_not_raised() -> None:
    desk = Desk()
    asked = signature()
    desk.arrive(asked)
    assert not desk.answer("s1", "yes")
    assert not asked._reply.done()  # pyright: ignore[reportPrivateUsage]


async def test_a_signature_answers_with_a_boolean() -> None:
    desk = Desk()
    asked = signature()
    desk.arrive(asked)
    assert desk.answer("s1", False)
    assert asked._reply.result() is False  # pyright: ignore[reportPrivateUsage]


async def test_clear_forgets_everything() -> None:
    desk = Desk()
    asked = question()
    desk.arrive(asked)
    desk.clear()
    assert desk.answering is None
    assert not desk.answer("a1", "late")
