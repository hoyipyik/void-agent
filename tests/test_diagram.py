"""The draw_diagram tool: chain-enabled, and nothing but its rules — the
diagram itself is the reply's own text."""

from __future__ import annotations

from void_agent import (
    Agent,
    AgentEvent,
    Answer,
    EventSender,
    PartsAccumulator,
    ScriptedLlm,
    ToolOutputAvailable,
    ToolOutputError,
    TurnResult,
    call,
    context_text,
    say,
)

REPLY = 'The order flow:\n\n```mermaid\nflowchart TD\n    A["Order"] --> B["Ship"]\n```'


async def run_collecting(agent: Agent) -> tuple[TurnResult, list[AgentEvent]]:
    events, queue = EventSender.channel(64)
    result = await agent.run({"q": "go"}, events)
    collected: list[AgentEvent] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    return result, collected


def drawing_agent(llm: ScriptedLlm) -> Agent:
    return Agent(llm, "drafter", "the agent under test").with_diagram()


def parts_of(events: list[AgentEvent]) -> list[dict[str, object]]:
    accumulator = PartsAccumulator()
    for event in events:
        accumulator.apply(event)
    return accumulator.into_parts()


async def test_with_diagram_enables_draw_diagram_and_the_rules_are_the_calls_result() -> None:
    agent = drawing_agent(ScriptedLlm([call("draw_diagram", {}), say(REPLY)]))
    result, events = await run_collecting(agent)
    assert result == Answer(REPLY)

    outputs = [event for event in events if isinstance(event, ToolOutputAvailable)]
    assert len(outputs) == 1
    assert "fenced code block tagged `mermaid`" in outputs[0].output


async def test_a_diagram_is_the_replys_own_text_and_leaves_no_part_of_its_own() -> None:
    agent = drawing_agent(ScriptedLlm([call("draw_diagram", {}), say(REPLY)]))
    _, events = await run_collecting(agent)

    parts = parts_of(events)
    assert [part["type"] for part in parts] == ["dynamic-tool", "text"]
    assert parts[1]["text"] == REPLY


async def test_the_rules_stay_in_the_conversation_for_the_next_turn() -> None:
    agent = drawing_agent(ScriptedLlm([call("draw_diagram", {}), say(REPLY)]))
    _, events = await run_collecting(agent)

    rendered = context_text(parts_of(events))
    assert rendered.startswith("[tool draw_diagram({}) → ")
    assert "fenced code block tagged `mermaid`" in rendered.splitlines()[0]


async def test_a_diagram_passed_as_an_argument_is_rejected_readably_and_the_model_recovers() -> (
    None
):
    agent = drawing_agent(
        ScriptedLlm(
            [
                call("draw_diagram", {"source": "flowchart TD\n    A --> B"}),
                say("recovered"),
            ]
        )
    )
    result, events = await run_collecting(agent)
    assert result == Answer("recovered")

    errors = [event for event in events if isinstance(event, ToolOutputError)]
    assert len(errors) == 1
    assert "invalid draw_diagram input" in errors[0].error_text


async def test_draw_diagram_is_unknown_without_with_diagram() -> None:
    agent = Agent(
        ScriptedLlm([call("draw_diagram", {}), say("gave up")]),
        "drafter",
        "the agent under test",
    )
    result, events = await run_collecting(agent)
    assert result == Answer("gave up")

    errors = [event for event in events if isinstance(event, ToolOutputError)]
    assert len(errors) == 1
    assert "unknown tool" in errors[0].error_text
