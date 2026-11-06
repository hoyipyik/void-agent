"""The person a case is asked by, played by a model when the case says
what they want (`brief`): they answer what the agent asks them — in place
when it calls `ask_user`, as their next message when it asks in words —
and say nothing when it asked them nothing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from tests.test_evals_case import MANAGER, run, setup

from void_agent import (
    EventSender,
    ModelStep,
    ScriptedLlm,
    SystemText,
    ToolSpec,
    TranscriptEntry,
    UserContent,
    UserText,
    call,
    say,
)
from void_agent.evals.case import Setup
from void_agent.evals.suite import Case, Turn

BRIEF = "You mean Contoso Ltd, the British one. You want this year's total."


def says(reply: str | None) -> ModelStep:
    """What the person's model hands in: words to say, or nothing."""
    return call("final_answer", {"reply": reply})


class Listening:
    """The person's model, keeping what it was shown each time."""

    def __init__(self, moves: Sequence[ModelStep]) -> None:
        self.shown: list[str] = []
        self._moves = ScriptedLlm(list(moves))

    async def step(
        self, transcript: Sequence[TranscriptEntry], tools: Sequence[ToolSpec], events: EventSender
    ) -> ModelStep:
        if not any(
            isinstance(entry, UserText) and "final_answer" in entry.text for entry in transcript
        ):
            self.shown.append("\n".join(_text(entry) for entry in transcript))
        return await self._moves.step(transcript, tools, events)


def _text(entry: TranscriptEntry) -> str:
    if isinstance(entry, SystemText | UserText):
        return entry.text
    if isinstance(entry, UserContent):
        return " ".join(getattr(part, "text", "") for part in entry.content)
    return ""


def briefed(*turns: str) -> Case:
    return Case("how-much", "manager", tuple(Turn(text) for text in turns), brief=BRIEF)


def with_person(the_setup: Setup, person: Any, rounds: int = 3) -> Setup:
    return replace(the_setup, person=person, person_rounds=rounds)


async def test_a_question_the_agent_asks_in_words_is_answered_by_the_person() -> None:
    person = Listening([says("the british one"), says(None)])
    record = await run(
        briefed("how much did contoso buy"),
        with_person(
            setup([say("Which Contoso do you mean?"), say("Contoso Ltd bought 10.")]), person
        ),
    )

    asked, answered = record["turns"]
    assert asked["answer"] == "Which Contoso do you mean?"
    assert answered["user"] == {"text": "the british one", "attach": [], "by": "person"}
    assert answered["answer"] == "Contoso Ltd bought 10."
    assert [move["reply"] for move in record["person"]["moves"]] == ["the british one", None]


async def test_the_person_says_nothing_when_the_agent_asked_nothing() -> None:
    person = Listening([says(None)])
    record = await run(
        briefed("how much did contoso buy"),
        with_person(setup([say("Contoso bought 10.")]), person),
    )

    assert len(record["turns"]) == 1
    assert record["person"]["moves"][0]["reply"] is None


async def test_the_person_answers_an_ask_user_question_in_place() -> None:
    person = Listening([says("the british one"), says(None)])
    record = await run(
        briefed("how much did contoso buy"),
        with_person(
            setup(
                [call("ask_user", {"question": "Which Contoso?"}, call_id="a1"), say("10 units.")]
            ),
            person,
        ),
    )

    (turn,) = record["turns"]
    assert turn["outcome"] == "answered"
    assert turn["asks"] == [{"question": "Which Contoso?", "answer": "the british one"}]
    returned = turn["steps"][1]["read"][1]["tool_returns"][0]
    assert returned["content"] == '{"answer":"the british one"}'


async def test_the_person_replies_no_more_often_than_the_rounds_allow() -> None:
    person = Listening([says("contoso"), says("contoso!!")])
    record = await run(
        briefed("how much"),
        with_person(
            setup([say("Which customer?"), say("Which Contoso?"), say("Which one?")]), person, 2
        ),
    )

    assert [turn["user"]["text"] for turn in record["turns"]] == [
        "how much",
        "contoso",
        "contoso!!",
    ]
    assert len(record["person"]["moves"]) == 2


async def test_the_person_is_shown_the_brief_and_the_words_but_never_the_tools() -> None:
    person = Listening([says(None)])
    await run(
        briefed("how much did contoso buy"),
        with_person(
            setup([call("search", {"q": "Contoso"}), say("Which Contoso do you mean?")]),
            person,
        ),
    )

    (shown,) = person.shown
    assert BRIEF in shown
    assert "how much did contoso buy" in shown
    assert "Which Contoso do you mean?" in shown
    assert "CUST-7" not in shown


async def test_the_person_asks_their_scripted_follow_up_after_answering() -> None:
    person = Listening([says("the british one"), says(None), says(None)])
    record = await run(
        briefed("how much did contoso buy", "and last year?"),
        with_person(
            setup(
                [say("Which Contoso?"), say("Contoso Ltd bought 10."), say("Nothing last year.")]
            ),
            person,
        ),
    )

    assert [(turn["user"]["text"], turn["user"]["by"]) for turn in record["turns"]] == [
        ("how much did contoso buy", "case"),
        ("the british one", "person"),
        ("and last year?", "case"),
    ]


async def test_a_case_without_a_brief_is_never_answered_by_the_person() -> None:
    person = Listening([])
    the_case = Case("how-much", "manager", (Turn("how much did contoso buy"),))

    record = await run(the_case, with_person(setup([say("Which Contoso?")]), person), MANAGER)

    assert len(record["turns"]) == 1
    assert "person" not in record


async def test_the_person_plays_who_the_suite_says_they_are() -> None:
    person = Listening([says(None)])
    await run(
        briefed("how much did contoso buy"),
        replace(with_person(setup([say("Contoso bought 10.")]), person), person_role="a buyer"),
    )

    assert "You play a buyer." in person.shown[0]


async def test_a_person_the_suite_says_nothing_of_is_told_no_trade() -> None:
    person = Listening([says(None)])
    await run(briefed("how much did contoso buy"), with_person(setup([say("10.")]), person))

    assert "You play a person who uses a chat assistant." in person.shown[0]
    assert "orders" not in person.shown[0]
