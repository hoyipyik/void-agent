"""The Responses API adapter's rendering and parsing contract, against a fake
client that replays canned stream events."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from openai import AsyncOpenAI
from tests.provider_fakes import OpenAiResponsesClient as FakeClient
from tests.provider_fakes import (
    function_call_item,
    message_item,
    output_text_delta,
    reasoning_item,
    response_done,
    response_failed,
    response_usage,
)

from void_agent import (
    AssistantStep,
    EventSender,
    Internal,
    ModelStep,
    SystemText,
    TextDelta,
    TextEnd,
    TextStart,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    Usage,
    UserText,
    tool_call,
)
from void_agent.providers.openai_responses import OpenAiResponsesLlm


def llm_on(fake: FakeClient, **kwargs: Any) -> OpenAiResponsesLlm:
    return OpenAiResponsesLlm("m", client=cast(AsyncOpenAI, fake), **kwargs)


def drain(queue: Any) -> list[Any]:
    events: list[Any] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def test_the_transcript_renders_to_input_items() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    step = ModelStep(text="calling", tool_calls=(tool_call("f", {"a": 1}, call_id="c1"),))
    await llm_on(fake).step(
        [
            SystemText("be brief"),
            UserText("hello"),
            AssistantStep(step),
            ToolReturns((ToolReturn(call_id="c1", name="f", content="42"),)),
        ],
        [ToolSpec(name="f", description="d", input_schema={"type": "object"})],
        EventSender(),
    )
    request = fake.requests[0]
    assert request["input"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "calling"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 1}'},
        {"type": "function_call_output", "call_id": "c1", "output": "42"},
    ]
    assert request["tools"] == [
        {
            "type": "function",
            "name": "f",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]


async def test_nothing_is_kept_on_openais_side_and_the_reasoning_comes_back_encrypted() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    await llm_on(fake).step([UserText("hi")], [], EventSender())
    request = fake.requests[0]
    assert request["store"] is False
    assert request["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in request
    assert "tools" not in request


async def test_text_streams_as_it_arrives_and_the_step_reads_the_final_output() -> None:
    fake = FakeClient(
        [output_text_delta("hel"), output_text_delta("lo"), response_done(message_item("hello"))]
    )
    sender, queue = EventSender.channel(64)
    step = await llm_on(fake).step([UserText("hi")], [], sender)
    assert step.text == "hello"
    events = drain(queue)
    assert isinstance(events[0], TextStart)
    assert [event.delta for event in events if isinstance(event, TextDelta)] == ["hel", "lo"]
    assert isinstance(events[-1], TextEnd)


async def test_function_calls_are_read_off_the_final_output() -> None:
    fake = FakeClient(
        [
            response_done(
                function_call_item("c9", "check", '{"sku":"W"}'),
                function_call_item("c10", "price", ""),
            )
        ]
    )
    step = await llm_on(fake).step([UserText("go")], [], EventSender())
    assert [(c.call_id, c.name, c.args) for c in step.tool_calls] == [
        ("c9", "check", {"sku": "W"}),
        ("c10", "price", {}),
    ]


async def test_arguments_that_do_not_parse_are_an_internal_error() -> None:
    fake = FakeClient([response_done(function_call_item("c1", "check", '{"sku":'))])
    with pytest.raises(Internal):
        await llm_on(fake).step([UserText("go")], [], EventSender())


async def test_the_output_replays_verbatim_its_reasoning_before_its_calls() -> None:
    reasoning = reasoning_item("gAAAA-opaque")
    call = function_call_item("c1", "check", '{"sku":"W"}')
    fake = FakeClient([response_done(reasoning, call)])
    llm = llm_on(fake)
    step = await llm.step([UserText("go")], [], EventSender())
    assert step.raw == [reasoning, call]

    await llm.step(
        [
            UserText("go"),
            AssistantStep(step),
            ToolReturns((ToolReturn(call_id="c1", name="check", content="60"),)),
        ],
        [],
        EventSender(),
    )
    assert fake.requests[1]["input"] == [
        {"role": "user", "content": "go"},
        reasoning,
        call,
        {"type": "function_call_output", "call_id": "c1", "output": "60"},
    ]


async def test_an_empty_output_carries_no_raw() -> None:
    fake = FakeClient([response_done()])
    step = await llm_on(fake).step([UserText("hi")], [], EventSender())
    assert step == ModelStep(text="", raw=None)


async def test_usage_counts_the_whole_prompt_with_its_cached_share() -> None:
    fake = FakeClient(
        [response_done(message_item("ok"), usage=response_usage(1200, 45, cached=900))]
    )
    step = await llm_on(fake).step([UserText("hi")], [], EventSender())
    assert step.usage == Usage(input=1200, output=45, cache_read=900)


async def test_a_response_without_usage_leaves_the_step_uncounted() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    step = await llm_on(fake).step([UserText("hi")], [], EventSender())
    assert step.usage is None


async def test_an_incomplete_response_keeps_what_was_said() -> None:
    fake = FakeClient([response_done(message_item("partial"), kind="response.incomplete")])
    step = await llm_on(fake).step([UserText("hi")], [], EventSender())
    assert step.text == "partial"


async def test_a_failed_response_is_an_internal_error() -> None:
    fake = FakeClient([response_failed("boom")])
    with pytest.raises(Internal):
        await llm_on(fake).step([UserText("hi")], [], EventSender())


async def test_an_error_event_is_an_internal_error() -> None:
    error = cast(Any, type("Event", (), {"type": "error", "code": "x", "message": "boom"})())
    fake = FakeClient([error])
    with pytest.raises(Internal):
        await llm_on(fake).step([UserText("hi")], [], EventSender())


async def test_a_stream_that_ends_before_its_response_is_an_internal_error() -> None:
    fake = FakeClient([output_text_delta("hal")])
    with pytest.raises(Internal):
        await llm_on(fake).step([UserText("hi")], [], EventSender())


async def test_extra_reaches_the_request() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    llm = llm_on(fake, extra={"reasoning": {"effort": "high"}})
    await llm.step([UserText("hi")], [], EventSender())
    assert fake.requests[0]["reasoning"] == {"effort": "high"}


async def test_the_stream_is_closed_after_the_step() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    await llm_on(fake).step([UserText("hi")], [], EventSender())
    assert fake.streams[0].closed is True


async def test_a_replayed_steps_usage_never_reaches_the_wire() -> None:
    fake = FakeClient([response_done(message_item("ok"))])
    costed = ModelStep(text="prior", usage=Usage(input=1200, output=45, cache_read=900))
    await llm_on(fake).step([UserText("hi"), AssistantStep(costed)], [], EventSender())
    wire = json.dumps(fake.requests[0])
    assert "1200" not in wire and "900" not in wire and "usage" not in wire
