"""The wire protocol is a contract: these tests pin exact JSON."""

from __future__ import annotations

import pytest

from void_agent import (
    AskAnswered,
    AskDropped,
    AskIssued,
    Call,
    Error,
    EventMode,
    EventSender,
    Finish,
    PlanUpdated,
    Progress,
    Start,
    StepStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
    ToolOutputError,
    Usage,
    UsageReported,
    to_wire,
)


def test_framing_events_wire_exactly() -> None:
    assert to_wire(Start(message_id="m1")) == {"type": "start", "messageId": "m1"}
    assert to_wire(Finish()) == {"type": "finish"}
    assert to_wire(Error(error_text="boom")) == {"type": "error", "errorText": "boom"}


def test_step_progress_wires_as_a_data_part() -> None:
    assert to_wire(StepStart(step=3)) == {"type": "data-step", "data": {"step": 3}}


def test_text_events_wire_exactly() -> None:
    assert to_wire(TextStart(id="t")) == {"type": "text-start", "id": "t"}
    assert to_wire(TextDelta(id="t", delta="hi")) == {
        "type": "text-delta",
        "id": "t",
        "delta": "hi",
    }
    assert to_wire(TextEnd(id="t")) == {"type": "text-end", "id": "t"}


def test_tool_events_wire_exactly_and_are_dynamic() -> None:
    assert to_wire(ToolInputStart(tool_call_id="c", tool_name="f")) == {
        "type": "tool-input-start",
        "toolCallId": "c",
        "toolName": "f",
        "dynamic": True,
    }
    assert to_wire(ToolInputAvailable(tool_call_id="c", tool_name="f", input={"a": 1})) == {
        "type": "tool-input-available",
        "toolCallId": "c",
        "toolName": "f",
        "input": {"a": 1},
        "dynamic": True,
    }
    assert to_wire(ToolOutputAvailable(tool_call_id="c", output="ok")) == {
        "type": "tool-output-available",
        "toolCallId": "c",
        "output": "ok",
        "dynamic": True,
    }
    assert to_wire(ToolOutputError(tool_call_id="c", error_text="no")) == {
        "type": "tool-output-error",
        "toolCallId": "c",
        "errorText": "no",
        "dynamic": True,
    }


def test_plan_and_ask_wire_as_data_parts() -> None:
    assert to_wire(PlanUpdated(items=[{"id": "1"}])) == {
        "type": "data-plan",
        "data": {"items": [{"id": "1"}]},
    }
    assert to_wire(
        AskIssued(ask_id="a1", kind="approval", question="ok?", options=("y", "n"), payload=None)
    ) == {
        "type": "data-ask",
        "data": {
            "askId": "a1",
            "kind": "approval",
            "question": "ok?",
            "options": ["y", "n"],
            "payload": None,
        },
    }


def test_progress_wires_with_its_kind() -> None:
    assert to_wire(Progress(kind="charts", data=[1])) == {"type": "data-charts", "data": [1]}


async def _sent_through(mode: EventMode, events: list[object]) -> list[object]:
    sender, queue = EventSender.channel(64)
    narrowed = sender.with_mode(mode)
    for event in events:
        await narrowed.send(event)  # type: ignore[arg-type]
    collected: list[object] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    return collected


async def test_activity_mode_hides_text_and_step_rhythm_but_passes_the_trace() -> None:
    passed = await _sent_through(
        EventMode.ACTIVITY,
        [
            StepStart(step=1),
            TextStart(id="t"),
            TextDelta(id="t", delta="x"),
            AskIssued(ask_id="a", kind="input", question="q?", options=None, payload=None),
            ToolInputStart(tool_call_id="c", tool_name="f"),
            ToolOutputAvailable(tool_call_id="c", output="data"),
            PlanUpdated(items=[]),
        ],
    )
    # The subtree's question passes: the person attending the root answers it.
    assert passed == [
        AskIssued(ask_id="a", kind="input", question="q?", options=None, payload=None),
        ToolInputStart(tool_call_id="c", tool_name="f"),
        ToolOutputAvailable(tool_call_id="c", output="data"),
        PlanUpdated(items=[]),
    ]


async def test_activity_redacted_blanks_tool_outputs() -> None:
    passed = await _sent_through(
        EventMode.ACTIVITY_REDACTED,
        [ToolOutputAvailable(tool_call_id="c", output="secret")],
    )
    assert passed == [ToolOutputAvailable(tool_call_id="c", output=None)]


async def test_nesting_keeps_the_strictest_mode() -> None:
    sender, queue = EventSender.channel(64)
    narrowed = sender.with_mode(EventMode.HIDDEN).with_mode(EventMode.ALL)
    await narrowed.send(ToolInputStart(tool_call_id="c", tool_name="f"))
    assert queue.empty()


async def test_a_bare_sender_is_a_silent_no_op() -> None:
    sender = EventSender()
    assert sender.is_live() is False
    await sender.send(Finish())  # must not raise


def test_a_channel_requires_positive_capacity() -> None:
    with pytest.raises(ValueError):
        EventSender.channel(0)


def test_reserved_data_kinds_cannot_be_forged_as_progress() -> None:
    for kind in (
        "step",
        "plan",
        "ask",
        "answer",
        "trigger",
        "cancelled",
        "error",
        "usage",
        "elapsed",
    ):
        with pytest.raises(ValueError, match="reserved"):
            Progress(kind=kind, data={})


async def test_tool_progress_with_an_ordinary_kind_still_passes() -> None:
    sender, queue = EventSender.channel(8)
    await sender.progress("gather", {"sku": "W"})
    assert queue.get_nowait() == Progress(kind="gather", data={"sku": "W"})


def test_an_ask_from_a_gate_wires_the_call_and_its_resolutions_wire_as_data() -> None:
    issued = AskIssued(
        ask_id="a1",
        kind="approval",
        question="ok?",
        options=None,
        payload=None,
        call=Call(tool="create_draft", input={"total": 270.0}),
    )
    assert to_wire(issued) == {
        "type": "data-ask",
        "data": {
            "askId": "a1",
            "kind": "approval",
            "question": "ok?",
            "options": None,
            "payload": None,
            "call": {"tool": "create_draft", "input": {"total": 270.0}},
        },
    }
    assert to_wire(AskAnswered(ask_id="a1", value=True)) == {
        "type": "data-answer",
        "data": {"askId": "a1", "value": True},
    }
    assert to_wire(AskAnswered(ask_id="a2", value="blue")) == {
        "type": "data-answer",
        "data": {"askId": "a2", "value": "blue"},
    }
    assert to_wire(AskDropped(ask_id="a1")) == {
        "type": "data-ask-dropped",
        "data": {"askId": "a1"},
    }


@pytest.mark.internals
def test_activity_mode_passes_a_subtrees_questions_so_the_attendant_can_answer() -> None:
    from void_agent.core.events.visibility import admit

    issued = AskIssued(ask_id="a1", kind="input", question="q?", options=None, payload=None)
    assert admit(EventMode.ACTIVITY, issued) == issued
    assert admit(EventMode.ACTIVITY, AskAnswered(ask_id="a1", value="v")) is not None
    assert admit(EventMode.ACTIVITY, AskDropped(ask_id="a1")) is not None
    assert admit(EventMode.HIDDEN, issued) is None


def test_usage_wires_as_a_data_part_with_the_cache_split() -> None:
    assert to_wire(UsageReported(Usage(input=1200, output=45, cache_read=900, cache_write=0))) == {
        "type": "data-usage",
        "data": {"input": 1200, "output": 45, "cacheRead": 900, "cacheWrite": 0},
    }


async def test_activity_mode_passes_a_subtrees_usage_so_the_root_account_is_whole() -> None:
    reported = UsageReported(Usage(input=10, output=2))
    assert await _sent_through(EventMode.ACTIVITY, [StepStart(step=1), reported]) == [reported]
