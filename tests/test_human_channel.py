"""The ready-made channel: every question the run tree asks arrives on a
queue as a `Question`, and its reply returns to the frame that asked."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from void_agent import (
    HUMAN,
    Agent,
    AgentEvent,
    Answer,
    Ask,
    AskAnswered,
    AskDropped,
    AskIssued,
    Call,
    EventSender,
    HumanChannel,
    ModelStep,
    Question,
    ScriptedLlm,
    Tool,
    TurnResult,
    call,
    say,
    tool,
    tool_call,
)


class DraftIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: float


class EchoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def make_draft(ran: list[DraftIn]) -> Tool:
    @tool(
        description="Creates a draft; large ones are signed by a human.",
        approval=lambda draft: f"total {draft.total} exceeds 100" if draft.total > 100 else None,
    )
    async def create_draft(input: DraftIn) -> dict[str, str]:
        ran.append(input)
        return {"draft_id": "D-1"}

    return create_draft


def start(
    agent: Agent, human: HumanChannel
) -> tuple[asyncio.Task[TurnResult], asyncio.Queue[AgentEvent]]:
    events, queue = EventSender.channel(64)
    return asyncio.create_task(agent.run({}, events, human=human)), queue


def ask_names(queue: asyncio.Queue[AgentEvent]) -> list[AgentEvent]:
    collected: list[AgentEvent] = []
    while not queue.empty():
        event = queue.get_nowait()
        if isinstance(event, AskIssued | AskAnswered | AskDropped):
            collected.append(event)
    return collected


def words_ask(ask_id: str, question: str) -> Ask:
    return Ask(ask_id=ask_id, kind="input", question=question, options=None, payload=None)


def signature_ask(ask_id: str, question: str) -> Ask:
    return Ask(
        ask_id=ask_id,
        kind="approval",
        question=question,
        options=None,
        payload=None,
        call=Call(tool="create_draft", input={"total": 270.0}),
    )


async def test_the_models_question_arrives_on_the_queue_and_its_reply_returns_in_place() -> None:
    llm = ScriptedLlm(
        [call("ask_user", {"question": "color?", "options": ["red", "blue"]}), say("blue it is")]
    )
    agent = Agent(llm, "subject", "the agent under test").tool(HUMAN)
    human, questions = HumanChannel.channel(4)
    run, events = start(agent, human)

    question = await questions.get()
    assert question.ask.question == "color?" and question.ask.options == ("red", "blue")
    assert not question.signature
    assert question.reply("blue") is True

    assert await run == Answer("blue it is")
    issued, answered = ask_names(events)
    assert isinstance(issued, AskIssued) and issued.ask_id == question.ask.ask_id
    assert answered == AskAnswered(ask_id=question.ask.ask_id, value="blue")


async def test_a_gates_question_is_signed_on_the_queue_and_the_call_runs_right_there() -> None:
    ran: list[DraftIn] = []
    llm = ScriptedLlm([call("create_draft", {"total": 270}), say("D-1 is ready")])
    agent = Agent(llm, "subject", "the agent under test").tool(make_draft(ran))
    human, questions = HumanChannel.channel(4)
    run, events = start(agent, human)

    question = await questions.get()
    assert question.signature
    assert question.ask.call == Call(tool="create_draft", input={"total": 270.0})
    assert ran == []
    assert question.reply(True) is True

    assert await run == Answer("D-1 is ready")
    assert ran == [DraftIn(total=270)]
    assert ask_names(events)[-1] == AskAnswered(ask_id=question.ask.ask_id, value=True)


async def test_a_reply_of_the_wrong_shape_is_a_programming_error() -> None:
    human, questions = HumanChannel.channel(4)

    approving = asyncio.create_task(human.approve(signature_ask("s", "sign?")))
    question = await questions.get()
    with pytest.raises(TypeError, match="a signature is replied to with a bool"):
        question.reply("approve")
    assert question.reply(False) is True and await approving is False

    answering = asyncio.create_task(human.answer(words_ask("w", "color?")))
    question = await questions.get()
    with pytest.raises(TypeError, match="words are replied to with a str"):
        question.reply(True)
    assert question.reply("blue") is True and await answering == "blue"


async def test_questions_in_flight_together_each_get_their_own_reply() -> None:
    human, questions = HumanChannel.channel(4)
    first = asyncio.create_task(human.answer(words_ask("a", "alpha?")))
    second = asyncio.create_task(human.approve(signature_ask("b", "beta?")))

    got = {question.ask.ask_id: question for question in [await questions.get() for _ in range(2)]}
    assert got["b"].reply(True) and got["a"].reply("alpha")
    assert await first == "alpha" and await second is True


async def test_no_reply_within_patience_drops_the_wait_and_a_late_reply_is_refused() -> None:
    llm = ScriptedLlm([call("ask_user", {"question": "sure?"})])
    agent = Agent(llm, "subject", "the agent under test").tool(HUMAN)
    human, questions = HumanChannel.channel(4, patience=0.01)
    run, events = start(agent, human)

    question = await questions.get()
    result = await run
    assert isinstance(result, Ask) and result.question == "sure?"
    assert [type(e).__name__ for e in ask_names(events)] == ["AskIssued", "AskDropped"]
    assert question.reply("late") is False


async def test_dropping_a_question_ends_the_run_with_the_card_open() -> None:
    ran: list[str] = []

    @tool(description="Records that it ran.")
    async def side_effect(input: EchoIn) -> str:
        ran.append(input.text)
        return "done"

    llm = ScriptedLlm(
        [
            ModelStep(
                text="",
                tool_calls=(
                    tool_call("ask_user", {"question": "sure?"}),
                    tool_call("side_effect", {"text": "boom"}),
                ),
            )
        ]
    )
    agent = Agent(llm, "subject", "the agent under test").tool(side_effect).tool(HUMAN)
    human, questions = HumanChannel.channel(4)
    run, events = start(agent, human)

    question = await questions.get()
    assert question.drop() is True
    result = await run
    assert isinstance(result, Ask) and result.question == "sure?"
    assert ran == []
    assert [type(e).__name__ for e in ask_names(events)] == ["AskIssued", "AskDropped"]
    assert question.reply("too late") is False


async def test_cancelling_the_run_cancels_the_wait_and_a_reply_after_it_is_refused() -> None:
    llm = ScriptedLlm([call("ask_user", {"question": "sure?"})])
    agent = Agent(llm, "subject", "the agent under test").tool(HUMAN)
    human, questions = HumanChannel.channel(4)
    run, _ = start(agent, human)

    question: Question = await questions.get()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert question.reply("late") is False


def test_a_channel_needs_room_for_at_least_one_question() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        HumanChannel.channel(0)
