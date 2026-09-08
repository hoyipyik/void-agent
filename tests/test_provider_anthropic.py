"""The Anthropic adapter's rendering and parsing contract, against a fake
client that replays canned stream events."""

from __future__ import annotations

from typing import Any, cast

from anthropic import AsyncAnthropic
from tests.provider_fakes import AnthropicClient as FakeClient
from tests.provider_fakes import final_message, text_block, text_delta, tool_use_block

from void_agent import (
    AssistantStep,
    EventSender,
    ModelStep,
    SystemText,
    TextDelta,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    UserText,
    tool_call,
)
from void_agent.providers.anthropic import AnthropicLlm


async def collect(queue: Any) -> list[Any]:
    events: list[Any] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def test_the_transcript_renders_to_system_messages_and_tool_results() -> None:
    fake = FakeClient([], final_message(text_block("ok")))
    llm = AnthropicLlm("claude-opus-5", client=cast(AsyncAnthropic, fake))
    step = ModelStep(text="", tool_calls=(tool_call("c1", {"a": 1}, call_id="c1"),))
    await llm.step(
        [
            SystemText("be brief"),
            UserText("hello"),
            AssistantStep(step),
            ToolReturns((ToolReturn(call_id="c1", name="c1", content="42"),)),
        ],
        [ToolSpec(name="f", description="d", input_schema={"type": "object"})],
        EventSender(),
    )
    request = fake.requests[0]
    assert request["system"] == "be brief"
    assert request["messages"][0] == {"role": "user", "content": "hello"}
    assert request["messages"][1]["role"] == "assistant"
    assert request["messages"][2]["content"][0]["type"] == "tool_result"
    assert request["messages"][2]["content"][0]["tool_use_id"] == "c1"
    assert request["tools"][0]["name"] == "f"


async def test_raw_content_replays_verbatim_on_continuation() -> None:
    fake = FakeClient([], final_message(text_block("ok")))
    llm = AnthropicLlm(client=cast(AsyncAnthropic, fake))
    verbatim = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    await llm.step(
        [UserText("hi"), AssistantStep(ModelStep(text="prior", raw=verbatim))],
        [],
        EventSender(),
    )
    assert fake.requests[0]["messages"][1]["content"] is verbatim


async def test_streamed_text_becomes_events_and_the_step_parses_blocks() -> None:
    fake = FakeClient(
        [text_delta("6"), text_delta("0 in stock")],
        final_message(text_block("60 in stock"), tool_use_block("t1", "check", {"sku": "W"})),
    )
    llm = AnthropicLlm(client=cast(AsyncAnthropic, fake))
    sender, queue = EventSender.channel(64)
    step = await llm.step([UserText("stock?")], [], sender)

    assert step.text == "60 in stock"
    assert (
        step.tool_calls == (tool_call("check", {"sku": "W"}, call_id="t1"),)
        or step.tool_calls[0].name == "check"
    )
    assert step.tool_calls[0].args == {"sku": "W"}
    assert step.raw is not None

    deltas = [event.delta for event in await collect(queue) if isinstance(event, TextDelta)]
    assert "".join(deltas) == "60 in stock"


async def test_an_empty_step_carries_no_raw_so_it_is_never_replayed() -> None:
    fake = FakeClient([], final_message())
    llm = AnthropicLlm(client=cast(AsyncAnthropic, fake))
    step = await llm.step([UserText("hi")], [], EventSender())
    assert step.text == ""
    assert step.raw is None
