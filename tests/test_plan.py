"""The plan tool: chain-enabled on the agent, last write wins, projection only."""

from __future__ import annotations

from void_agent import (
    Agent,
    AgentEvent,
    Answer,
    EventSender,
    PlanUpdated,
    ScriptedLlm,
    ToolOutputError,
    TurnResult,
    call,
    say,
)


async def run_collecting(agent: Agent) -> tuple[TurnResult, list[AgentEvent]]:
    events, queue = EventSender.channel(64)
    result = await agent.run({"q": "go"}, events)
    collected: list[AgentEvent] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    return result, collected


def plan_agent(llm: ScriptedLlm) -> Agent:
    return Agent(llm, "planner", "the agent under test").with_plan()


async def test_with_plan_enables_update_plan_and_the_last_write_wins() -> None:
    agent = plan_agent(
        ScriptedLlm(
            [
                call(
                    "update_plan",
                    {"items": [{"id": "1", "title": "look up stock", "status": "in_progress"}]},
                ),
                call(
                    "update_plan",
                    {
                        "items": [
                            {
                                "id": "1",
                                "title": "look up stock",
                                "status": "completed",
                                "note": "60 left",
                            }
                        ]
                    },
                ),
                say("done"),
            ]
        )
    )
    result, events = await run_collecting(agent)
    assert result == Answer("done")

    plans = [event for event in events if isinstance(event, PlanUpdated)]
    assert len(plans) == 2
    assert plans[1].items[0]["status"] == "completed"
    assert plans[1].items[0]["note"] == "60 left"


async def test_an_unknown_status_is_rejected_readably_and_the_model_recovers() -> None:
    agent = plan_agent(
        ScriptedLlm(
            [
                call("update_plan", {"items": [{"id": "1", "title": "x", "status": "done"}]}),
                say("recovered"),
            ]
        )
    )
    result, events = await run_collecting(agent)
    assert result == Answer("recovered")

    errors = [event for event in events if isinstance(event, ToolOutputError)]
    assert len(errors) == 1
    assert "invalid update_plan input" in errors[0].error_text


async def test_update_plan_is_unknown_without_with_plan() -> None:
    agent = Agent(
        ScriptedLlm(
            [
                call("update_plan", {"items": []}),
                say("gave up"),
            ]
        ),
        "planner",
        "the agent under test",
    )
    result, events = await run_collecting(agent)
    assert result == Answer("gave up")

    errors = [event for event in events if isinstance(event, ToolOutputError)]
    assert len(errors) == 1
    assert "unknown tool" in errors[0].error_text
