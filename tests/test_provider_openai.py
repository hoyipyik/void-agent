"""The OpenAI-compatible adapter's rendering and parsing contract, against a
fake client that replays canned stream chunks."""

from __future__ import annotations

import json
from typing import Any, cast

from openai import AsyncOpenAI
from tests.provider_fakes import OpenAiClient as FakeClient
from tests.provider_fakes import call_delta, chunk, usage_chunk

from void_agent import (
    AssistantStep,
    EventSender,
    ModelStep,
    SystemText,
    TextDelta,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    Usage,
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


async def test_usage_is_asked_for_and_read_off_the_last_chunk() -> None:
    fake = FakeClient([chunk(content="ok"), usage_chunk(1200, 45, cached=900)])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    step = await llm.step([UserText("hi")], [], EventSender())
    assert fake.requests[0]["stream_options"] == {"include_usage": True}
    assert step.text == "ok"
    assert step.usage == Usage(input=1200, output=45, cache_read=900)


async def test_usage_without_a_cache_split_reads_plainly() -> None:
    fake = FakeClient([chunk(content="ok"), usage_chunk(30, 45)])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    step = await llm.step([UserText("hi")], [], EventSender())
    assert step.usage == Usage(input=30, output=45)


async def test_a_server_that_reports_no_usage_leaves_the_step_uncounted() -> None:
    fake = FakeClient([chunk(content="ok")])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    step = await llm.step([UserText("hi")], [], EventSender())
    assert step.usage is None


async def test_extra_can_take_the_usage_request_back() -> None:
    fake = FakeClient([chunk(content="ok")])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake), extra={"stream_options": None})
    await llm.step([UserText("hi")], [], EventSender())
    assert fake.requests[0]["stream_options"] is None


async def test_a_replayed_steps_usage_never_reaches_the_wire() -> None:
    fake = FakeClient([chunk(content="ok")])
    llm = OpenAiLlm("m", client=cast(AsyncOpenAI, fake))
    costed = ModelStep(text="prior", usage=Usage(input=1200, output=45, cache_read=900))
    await llm.step([UserText("hi"), AssistantStep(costed)], [], EventSender())
    wire = json.dumps(
        {key: value for key, value in fake.requests[0].items() if key != "stream_options"}
    )
    assert "1200" not in wire and "900" not in wire and "usage" not in wire
