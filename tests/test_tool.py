"""The tool boundary: validation in, serialization out, errors classified."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel, ConfigDict

from void_agent import (
    AskAnswered,
    AskIssued,
    Call,
    EventSender,
    Internal,
    Rejected,
    ScriptedHuman,
    Tool,
    Unanswered,
    attended,
    tool,
)


class AddIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: int
    b: int


async def test_valid_input_is_validated_and_output_serialized() -> None:
    @tool(description="Adds two numbers.")
    async def add(input: AddIn) -> dict[str, int]:
        return {"total": input.a + input.b}

    assert await add.invoke({"a": 2, "b": 3}, EventSender()) == {"total": 5}


async def test_invalid_input_is_rejected_with_the_tool_name() -> None:
    @tool(description="Adds two numbers.")
    async def add(input: AddIn) -> int:
        return input.a + input.b

    with pytest.raises(Rejected, match="invalid add input"):
        await add.invoke({"a": "NaN", "b": 1}, EventSender())


async def test_a_handlers_rejection_passes_through_verbatim() -> None:
    @tool(description="Refuses.")
    async def guard(input: AddIn) -> int:
        raise Rejected("limit exceeded")

    with pytest.raises(Rejected, match="limit exceeded"):
        await guard.invoke({"a": 1, "b": 1}, EventSender())


async def test_an_unexpected_failure_becomes_internal() -> None:
    @tool(description="Crashes.")
    async def crash(input: AddIn) -> int:
        raise ValueError("db exploded")

    with pytest.raises(Internal):
        await crash.invoke({"a": 1, "b": 1}, EventSender())


async def test_a_handler_may_ask_for_the_event_sender() -> None:
    received: list[EventSender] = []

    @tool(description="Wants events.")
    async def wants(input: AddIn, events: EventSender) -> int:
        received.append(events)
        return 0

    sender = EventSender()
    await wants.invoke({"a": 1, "b": 1}, sender)
    assert received == [sender]


async def test_a_keyword_only_event_sender_is_injected_by_its_declared_name() -> None:
    received: list[EventSender] = []

    @tool(description="Wants keyword events.")
    async def wants(input: AddIn, *, progress: EventSender) -> int:
        received.append(progress)
        return input.a + input.b

    sender = EventSender()
    assert await wants.invoke({"a": 1, "b": 2}, sender) == 3
    assert received == [sender]


async def test_a_positional_only_event_sender_is_injected() -> None:
    received: list[EventSender] = []

    @tool(description="Wants positional events.")
    async def wants(input: AddIn, progress: EventSender, /) -> int:
        received.append(progress)
        return input.a + input.b

    sender = EventSender()
    assert await wants.invoke({"a": 1, "b": 2}, sender) == 3
    assert received == [sender]


def test_unsupported_handler_signatures_fail_at_registration() -> None:
    async def business_parameter(input: AddIn, limit: int) -> None:
        pass

    async def duplicate_events(input: AddIn, first: EventSender, second: EventSender) -> None:
        pass

    async def keyword_input(*, input: AddIn) -> None:
        pass

    async def variadic_input(*input: AddIn) -> None:
        pass

    async def variadic_events(input: AddIn, *events: EventSender) -> None:
        pass

    async def keyword_variadic_events(input: AddIn, **events: EventSender) -> None:
        pass

    async def missing_event_annotation(input: AddIn, events) -> None:  # type: ignore[no-untyped-def]
        pass

    handlers: list[Callable[..., Awaitable[None]]] = [
        business_parameter,
        duplicate_events,
        keyword_input,
        variadic_input,
        variadic_events,
        keyword_variadic_events,
        missing_event_annotation,  # pyright: ignore[reportUnknownVariableType]
    ]
    for handler in handlers:
        with pytest.raises(TypeError, match=f"tool `{handler.__name__}`"):
            Tool(name=handler.__name__, description="Unsupported signature.", handler=handler)


def test_the_description_is_mandatory() -> None:
    async def nameless(input: AddIn) -> int:
        return 0

    with pytest.raises(ValueError, match="description"):
        Tool(name="x", description="  ", handler=nameless)


def test_an_unannotated_input_is_a_construction_error() -> None:
    async def untyped(input) -> int:  # type: ignore[no-untyped-def]
        return 0

    with pytest.raises(TypeError, match="annotate"):
        Tool(name="untyped", description="broken", handler=untyped)  # pyright: ignore[reportUnknownArgumentType]


def test_the_docstring_becomes_the_description() -> None:
    @tool()
    async def documented(input: AddIn) -> int:
        """Adds a and b."""
        return input.a + input.b

    assert documented.description == "Adds a and b."
    assert documented.input_schema["additionalProperties"] is False


async def test_a_tool_may_declare_a_schema_it_did_not_derive() -> None:
    """A tool whose contract comes from elsewhere — an MCP server's
    descriptor — shows the model that schema, and still validates, gates
    and serializes like any other."""
    given = {"type": "object", "properties": {"path": {"type": "string"}}}

    async def passthrough(input: dict[str, object]) -> dict[str, object]:
        return input

    declared = Tool(
        name="write_file",
        description="writes a file",
        handler=passthrough,
        input_schema=given,
    )
    assert declared.input_schema == given
    assert await declared.invoke({"path": "/a"}, EventSender()) == {"path": "/a"}


# ── the approval gate: the tool asks the person itself ───────────────────


def limit_approval(input: AddIn) -> str | None:
    return f"sum {input.a + input.b} exceeds the auto-limit" if input.a + input.b > 10 else None


def gated_add(ran: list[AddIn]) -> Tool:
    @tool(description="Adds, but big sums are signed by a human.", approval=limit_approval)
    async def add(input: AddIn) -> int:
        ran.append(input)
        return input.a + input.b

    return add


async def test_a_gate_asks_the_attendant_and_an_approval_runs_the_call() -> None:
    ran: list[AddIn] = []
    human = ScriptedHuman([True])
    events, queue = EventSender.channel(8)
    with attended(human):
        assert await gated_add(ran).invoke({"a": 7, "b": 8}, events) == 15
    assert ran == [AddIn(a=7, b=8)]
    # The card carries the gate's words and the exact call — runtime-written.
    [ask] = human.asked
    assert ask.kind == "approval" and ask.question == "sum 15 exceeds the auto-limit"
    assert ask.call == Call(tool="add", input={"a": 7, "b": 8})
    issued, answered = queue.get_nowait(), queue.get_nowait()
    assert (
        isinstance(issued, AskIssued) and issued.ask_id == ask.ask_id and issued.call == ask.call
    )
    assert answered == AskAnswered(ask_id=ask.ask_id, value=True)


async def test_a_declined_call_is_a_readable_rejection() -> None:
    ran: list[AddIn] = []
    with attended(ScriptedHuman([False])), pytest.raises(Rejected, match="declined add"):
        await gated_add(ran).invoke({"a": 7, "b": 8}, EventSender())
    assert ran == []


async def test_an_unanswered_gate_unwinds_with_the_question() -> None:
    ran: list[AddIn] = []
    events, queue = EventSender.channel(8)
    with attended(ScriptedHuman([])), pytest.raises(Unanswered) as unanswered:
        await gated_add(ran).invoke({"a": 7, "b": 8}, events)
    assert ran == []
    assert unanswered.value.ask.call == Call(tool="add", input={"a": 7, "b": 8})
    assert [type(e).__name__ for e in (queue.get_nowait(), queue.get_nowait())] == [
        "AskIssued",
        "AskDropped",
    ]


async def test_with_nobody_attending_the_card_stays_open() -> None:
    events, queue = EventSender.channel(8)
    with pytest.raises(Unanswered):
        await gated_add([]).invoke({"a": 7, "b": 8}, events)
    assert isinstance(queue.get_nowait(), AskIssued) and queue.empty()  # no AskDropped


async def test_an_approval_that_returns_none_lets_the_call_run() -> None:
    assert await gated_add([]).invoke({"a": 1, "b": 2}, EventSender()) == 3


async def test_an_approval_may_be_async() -> None:
    async def needs_signature(input: AddIn) -> str | None:
        return "checked against the ledger"

    @tool(description="Adds.", approval=needs_signature)
    async def add(input: AddIn) -> int:
        return input.a + input.b

    with pytest.raises(Unanswered, match="ledger"):
        await add.invoke({"a": 1, "b": 2}, EventSender())


async def test_an_approval_only_sees_valid_input() -> None:
    seen: list[AddIn] = []

    def needs_signature(input: AddIn) -> str | None:
        seen.append(input)
        return "signed"

    @tool(description="Adds.", approval=needs_signature)
    async def add(input: AddIn) -> int:
        return 0

    with pytest.raises(Rejected, match="invalid add input"):
        await add.invoke({"a": "NaN", "b": 1}, EventSender())
    assert seen == []


async def test_a_crashing_approval_is_internal() -> None:
    def needs_signature(input: AddIn) -> str | None:
        raise RuntimeError("ledger down")

    @tool(description="Adds.", approval=needs_signature)
    async def add(input: AddIn) -> int:
        return 0

    with pytest.raises(Internal):
        await add.invoke({"a": 1, "b": 1}, EventSender())


async def test_a_gated_tool_inside_a_workflow_asks_the_person_directly() -> None:
    """No relay: the inner gate consults the attendant itself; the workflow
    just sees the value. Nobody answering unwinds through the workflow —
    even through an `except Exception` it did not mean for this."""
    ran: list[AddIn] = []
    add = gated_add(ran)

    @tool(description="Adds twice via the gated tool.")
    async def add_twice(input: AddIn, events: EventSender) -> int:
        try:
            return int(await add.invoke({"a": input.a, "b": input.b}, events)) * 2
        except Exception:  # a broad net a workflow author might cast
            return -1

    with attended(ScriptedHuman([True])):
        assert await add_twice.invoke({"a": 7, "b": 8}, EventSender()) == 30
    assert ran == [AddIn(a=7, b=8)]

    with pytest.raises(Unanswered) as unanswered:  # nobody attends
        await add_twice.invoke({"a": 7, "b": 8}, EventSender())
    assert unanswered.value.ask.call is not None and unanswered.value.ask.call.tool == "add"
