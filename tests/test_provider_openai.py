"""The OpenAI-compatible adapter's rendering and parsing contract, against a
fake client that replays canned stream chunks."""

from __future__ import annotations

from typing import Any, cast

from openai import AsyncOpenAI
from tests.provider_fakes import OpenAiClient as FakeClient
from tests.provider_fakes import call_delta, chunk

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
from void_agent.providers.openai import OpenAiLlm


async def collect_deltas(queue: Any) -> str:
    deltas: list[str] = []
    while not queue.empty():
        event = queue.get_nowait()
        if isinstance(event, TextDelta):
            deltas.append(event.delta)
    return "".join(deltas)


async def test_the_transcript_renders_to_chat_messages() -> None:
    fake = FakeClient([chunk(content="ok")])
    llm = OpenAiLlm("some-model", client=cast(AsyncOpenAI, fake))
    step = ModelStep(text="calling", tool_calls=(tool_call("f", {"a": 1}, call_id="c1"),))
    await llm.step(
        [
            SystemText("be brief"),
            UserText("hello"),
            AssistantStep(step),
            ToolReturns((ToolReturn(call_id="c1", name="f", content="42"),)),
        ],
        [ToolSpec(name="f", description="d", input_schema={"type": "object"})],
        EventSender(),
    )
    messages = fake.requests[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "f"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "42"}
    assert fake.requests[0]["tools"][0]["function"]["name"] == "f"


async def test_streamed_tool_calls_assemble_across_chunks() -> None:
    fake = FakeClient(
        [
            chunk(tool_calls=[call_delta(0, call_id="c9", name="check", arguments='{"sk')]),
            chunk(tool_calls=[call_delta(0, arguments='u":"W"}')]),
        ]
    )
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    step = await llm.step([UserText("go")], [], EventSender())
    assert step.tool_calls[0].call_id == "c9"
    assert step.tool_calls[0].name == "check"
    assert step.tool_calls[0].args == {"sku": "W"}


async def test_plain_text_streams_through() -> None:
    fake = FakeClient([chunk(content="hel"), chunk(content="lo")])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    sender, queue = EventSender.channel(64)
    step = await llm.step(
        [UserText("hi")],
        [ToolSpec(name="f", description="d", input_schema={})],
        sender,
    )
    assert step.text == "hello"
    assert await collect_deltas(queue) == "hello"


async def test_a_tool_argument_echo_is_held_back_and_suppressed() -> None:
    fake = FakeClient(
        [
            chunk(content='{"sku":"W"}'),
            chunk(tool_calls=[call_delta(0, call_id="c1", name="check", arguments='{"sku":"W"}')]),
        ]
    )
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    sender, queue = EventSender.channel(64)
    step = await llm.step(
        [UserText("go")],
        [ToolSpec(name="check", description="d", input_schema={})],
        sender,
    )
    assert step.text == ""
    assert await collect_deltas(queue) == ""


async def test_a_json_shaped_answer_with_no_tools_still_streams() -> None:
    fake = FakeClient([chunk(content='{"answer":42}')])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    sender, queue = EventSender.channel(64)
    step = await llm.step([UserText("json please")], [], sender)
    assert step.text == '{"answer":42}'
    assert await collect_deltas(queue) == '{"answer":42}'


async def test_the_stream_is_closed_after_the_step() -> None:
    fake = FakeClient([chunk(content="ok")])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    await llm.step([UserText("hi")], [], EventSender())
    assert fake.streams[0].closed is True


async def test_real_words_after_an_argument_echo_survive() -> None:
    fake = FakeClient(
        [
            chunk(content='{"sku":"W"}\nOnly 60 in stock, checking pricing now.'),
            chunk(tool_calls=[call_delta(0, call_id="c1", name="check", arguments='{"sku":"W"}')]),
        ]
    )
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    sender, queue = EventSender.channel(64)
    step = await llm.step(
        [UserText("go")],
        [ToolSpec(name="check", description="d", input_schema={})],
        sender,
    )
    assert step.text == "Only 60 in stock, checking pricing now."
    assert await collect_deltas(queue) == "Only 60 in stock, checking pricing now."


async def test_index_reuse_by_compatible_servers_yields_separate_calls() -> None:
    fake = FakeClient(
        [
            chunk(tool_calls=[call_delta(0, call_id="c1", name="left", arguments='{"a":1}')]),
            chunk(tool_calls=[call_delta(0, call_id="c2", name="right", arguments='{"b":2}')]),
        ]
    )
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    step = await llm.step([UserText("go")], [], EventSender())
    assert [(c.call_id, c.name, c.args) for c in step.tool_calls] == [
        ("c1", "left", {"a": 1}),
        ("c2", "right", {"b": 2}),
    ]
