"""The reflect tool: chain-enabled, a durable projection of self-assessment."""

from __future__ import annotations

import pytest

from void_agent import (
    Agent,
    AgentEvent,
    Answer,
    EventSender,
    PartsAccumulator,
    Progress,
    ReflectionMade,
    ScriptedLlm,
    ToolOutputError,
    TurnResult,
    call,
    context_text,
    say,
)


async def run_collecting(agent: Agent) -> tuple[TurnResult, list[AgentEvent]]:
    events, queue = EventSender.channel(64)
    result = await agent.run({"q": "go"}, events)
    collected: list[AgentEvent] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    return result, collected


def reflective_agent(llm: ScriptedLlm) -> Agent:
    return Agent(llm, "subject", "the agent under test").with_reflection()


async def test_with_reflection_enables_reflect_and_it_survives_into_context() -> None:
    agent = reflective_agent(
        ScriptedLlm(
            [
                call(
                    "reflect",
                    {
                        "facts": "create_pi_draft was rejected: unknown SKU",
                        "verdict": "adjust",
                        "problems": ["the SKU came from my guess, not from the catalog"],
                        "adjustment": "resolve the SKU via the catalog first",
                    },
                ),
                say("done"),
            ]
        )
    )
    result, events = await run_collecting(agent)
    assert result == Answer("done")

    reflections = [event for event in events if isinstance(event, ReflectionMade)]
    assert len(reflections) == 1
    assert reflections[0].verdict == "adjust"

    accumulator = PartsAccumulator()
    accumulator.apply(reflections[0])
    parts = accumulator.into_parts()
    assert parts[0]["type"] == "data-reflection"

    rendered = context_text(parts)
    assert rendered.startswith("[reflection (adjust): create_pi_draft was rejected")
    assert "problems: the SKU came from my guess" in rendered
    assert "next: resolve the SKU via the catalog first" in rendered


async def test_an_unknown_verdict_is_rejected_readably_and_the_model_recovers() -> None:
    agent = reflective_agent(
        ScriptedLlm(
            [
                call("reflect", {"facts": "x", "verdict": "fine"}),
                say("recovered"),
            ]
        )
    )
    result, events = await run_collecting(agent)
    assert result == Answer("recovered")

    errors = [event for event in events if isinstance(event, ToolOutputError)]
    assert len(errors) == 1
    assert "invalid reflect input" in errors[0].error_text


def test_reflection_is_a_reserved_kind_tools_cannot_forge() -> None:
    with pytest.raises(ValueError, match="reserved"):
        Progress(kind="reflection", data={})
