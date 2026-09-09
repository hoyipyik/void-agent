"""One turn drained on the loop: the stream folded into parts, the
questions handed out as they come, a stop and a readable failure each
written into the parts as a marker."""

from __future__ import annotations

import asyncio
from typing import Any

from cli.session.runner import Turn
from pydantic import BaseModel, ConfigDict

from void_agent import (
    HUMAN,
    Agent,
    AgentEvent,
    ModelStep,
    Question,
    ScriptedLlm,
    ScriptedStep,
    ToolInputStart,
    call,
    say,
    tool,
    tool_call,
)


class Nothing(BaseModel):
    model_config = ConfigDict(extra="forbid")


@tool(description="never returns")
async def forever(input: Nothing) -> dict[str, str]:
    await asyncio.sleep(3600)
    return {}


@tool(description="returns at once")
async def ping(input: Nothing) -> dict[str, str]:
    return {"pong": "yes"}


def agent_with(script: list[ScriptedStep], *, max_steps: int | None = None) -> Agent:
    agent = Agent(ScriptedLlm(script), "void", "test").prompt(lambda history: list(history))
    agent = agent.tool(HUMAN).tool(forever).tool(ping)
    return agent.with_max_steps(max_steps) if max_steps is not None else agent


async def nothing(event: AgentEvent, parts: list[dict[str, Any]]) -> None:
    pass


def unasked(question: Question) -> None:
    raise AssertionError("no question was expected")


async def test_a_turn_folds_its_stream_into_parts() -> None:
    seen: list[tuple[AgentEvent, int]] = []

    async def on_event(event: AgentEvent, parts: list[dict[str, Any]]) -> None:
        seen.append((event, len(parts)))

    turn = Turn(agent_with([say("hello")]), [])
    parts = await turn.drain(on_event, unasked)
    assert parts == [{"type": "text", "text": "hello"}]
    assert seen and seen[-1][1] == 1
    assert not turn.running


async def test_a_question_is_handed_out_and_its_answer_returns_in_place() -> None:
    script: list[ScriptedStep] = [
        ModelStep(
            text="",
            tool_calls=(tool_call("ask_user", {"kind": "input", "question": "Which colour?"}),),
        ),
        say("blue it is"),
    ]
    asked: list[Question] = []

    def on_question(question: Question) -> None:
        asked.append(question)
        assert question.reply("blue")

    parts = await Turn(agent_with(script), []).drain(nothing, on_question)
    assert [question.ask.question for question in asked] == ["Which colour?"]
    types = [part["type"] for part in parts]
    assert "data-ask" in types and "data-answer" in types
    assert parts[-1] == {"type": "text", "text": "blue it is"}


async def test_a_stop_ends_the_turn_marked_cancelled() -> None:
    started = asyncio.Event()

    async def on_event(event: AgentEvent, parts: list[dict[str, Any]]) -> None:
        if isinstance(event, ToolInputStart):
            started.set()

    turn = Turn(agent_with([call("forever", {})]), [])
    draining = asyncio.create_task(turn.drain(on_event, unasked))
    await asyncio.wait_for(started.wait(), 5)
    assert turn.running
    turn.cancel()
    parts = await asyncio.wait_for(draining, 5)
    assert parts[-1] == {"type": "data-cancelled", "data": {}}
    assert parts[0]["type"] == "dynamic-tool"


async def test_a_readable_failure_becomes_a_data_error_part() -> None:
    script: list[ScriptedStep] = [call("ping", {}), call("ping", {}), say("never")]
    parts = await Turn(agent_with(script, max_steps=1), []).drain(nothing, unasked)
    assert parts[-1]["type"] == "data-error"
    assert "step limit" in parts[-1]["data"]["text"]
