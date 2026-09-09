"""The loop's observable behaviors, scripted through the `Llm` seam."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from void_agent import (
    HELD_TOOL_OUTPUT,
    HUMAN,
    Agent,
    AgentEvent,
    Answer,
    Ask,
    AskIssued,
    Attendant,
    Call,
    EventMode,
    EventSender,
    Exhausted,
    Internal,
    ModelStep,
    Rejected,
    ScriptedHuman,
    ScriptedLlm,
    ScriptedStep,
    StepStart,
    TextDelta,
    Tool,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
    ToolOutputError,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    TurnResult,
    Usage,
    UsageReported,
    call,
    say,
    tool,
    tool_call,
)

TOOL_LIFECYCLE_EVENTS = (ToolInputStart, ToolInputAvailable, ToolOutputAvailable, ToolOutputError)


class EchoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SumOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: int


@tool(description="Echoes the text back.")
async def echo(input: EchoIn) -> str:
    return input.text


class CapturingLlm(ScriptedLlm):
    """The scripted seam plus a copy of every transcript it was shown."""

    def __init__(self, steps: Sequence[ScriptedStep]) -> None:
        super().__init__(steps)
        self.transcripts: list[list[TranscriptEntry]] = []
        self.advertised: list[list[ToolSpec]] = []

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep:
        self.transcripts.append(list(transcript))
        self.advertised.append(list(tools))
        return await super().step(transcript, tools, events)


async def run_collecting(
    agent: Agent, input: Any, human: Attendant | None = None
) -> tuple[TurnResult, list[AgentEvent]]:
    events, queue = EventSender.channel(512)
    result = await agent.run(input, events, human=human)
    collected: list[AgentEvent] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    return result, collected


def make_agent(llm: ScriptedLlm) -> Agent:
    return Agent(llm, "subject", "the agent under test")


async def test_a_text_answer_finishes_the_turn() -> None:
    agent = make_agent(ScriptedLlm([say("hello there")]))
    result = await agent.run({"q": "hi"})
    assert result == Answer("hello there")


async def test_a_typed_submission_validates_and_finishes() -> None:
    agent = make_agent(ScriptedLlm([call("final_answer", {"total": 5})])).output(SumOut)
    result = await agent.run({})
    assert isinstance(result, Answer)
    assert result.value == SumOut(total=5)


async def test_output_instructions_reach_the_model_without_replacing_submission_rules() -> None:
    llm = CapturingLlm(
        [call("final_answer", {"total": "invalid"}), call("final_answer", {"total": 5})]
    )
    instructions = "Submit only the verified total; do not estimate missing amounts."
    agent = make_agent(llm).output(SumOut, instructions=instructions)
    result, events = await run_collecting(agent, {})
    assert result == Answer(SumOut(total=5))
    for specs in llm.advertised:
        [submission] = specs
        assert submission.name == "final_answer"
        assert instructions in submission.description
        assert "only way to finish" in submission.description
        assert submission.input_schema == SumOut.model_json_schema()
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content.startswith("ERROR: invalid final_answer")
    assert not any(isinstance(event, TOOL_LIFECYCLE_EVENTS) for event in events)


async def test_reconfiguring_output_replaces_both_schema_and_instructions() -> None:
    llm = CapturingLlm([call("final_answer", {"text": "ready"})])
    agent = make_agent(llm).output(SumOut, instructions="Old sum instructions.")
    agent.output(EchoIn)
    assert await agent.run({}) == Answer(EchoIn(text="ready"))
    [submission] = llm.advertised[0]
    assert submission.input_schema == EchoIn.model_json_schema()
    assert "Old sum instructions." not in submission.description


async def test_parent_and_child_keep_their_own_output_instructions() -> None:
    child_llm = CapturingLlm([call("final_answer", {"total": 7})])
    child = Agent(child_llm, "helper", "Adds.", input_type=EchoIn).output(
        SumOut, instructions="Child: submit the calculated sum."
    )
    parent_llm = CapturingLlm(
        [call("helper", {"text": "add"}), call("final_answer", {"text": "seven"})]
    )
    parent = (
        make_agent(parent_llm)
        .tool(child)
        .output(EchoIn, instructions="Parent: describe the result in words.")
    )
    assert await parent.run({}) == Answer(EchoIn(text="seven"))
    [child_submission] = child_llm.advertised[0]
    assert "Child:" in child_submission.description
    assert "Parent:" not in child_submission.description
    specs = {spec.name: spec for spec in parent_llm.advertised[0]}
    assert specs["helper"].input_schema == EchoIn.model_json_schema()
    assert "Parent:" in specs["final_answer"].description
    assert "Child:" not in specs["final_answer"].description
    returns = next(entry for entry in parent_llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content == '{"total":7}'


async def test_final_answer_is_unknown_without_an_output_type() -> None:
    llm = CapturingLlm([call("final_answer", {"total": 1}), say("done")])
    assert await make_agent(llm).run({}) == Answer("done")
    assert all(spec.name != "final_answer" for spec in llm.advertised[0])
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content == "ERROR: unknown tool: final_answer"


async def test_a_shared_tools_event_visibility_belongs_to_each_registration() -> None:
    visible_llm = CapturingLlm([call("echo", {"text": "visible"}), say("done")])
    hidden_llm = CapturingLlm([call("echo", {"text": "hidden"}), say("done")])
    visible = make_agent(visible_llm).tool(echo, mode=EventMode.ALL)
    hidden = make_agent(hidden_llm).tool(echo, mode=EventMode.HIDDEN)
    (_, visible_events), (_, hidden_events) = await asyncio.gather(
        run_collecting(visible, {}), run_collecting(hidden, {})
    )
    assert any(isinstance(event, ToolOutputAvailable) for event in visible_events)
    assert not any(isinstance(event, TOOL_LIFECYCLE_EVENTS) for event in hidden_events)
    for llm, expected in ((visible_llm, "visible"), (hidden_llm, "hidden")):
        assert [spec.name for spec in llm.advertised[0]] == ["echo"]
        returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
        assert returns.returns[0].content == expected


async def test_a_parents_human_setting_does_not_enable_ask_user_for_its_child() -> None:
    child_llm = CapturingLlm([call("ask_user", {"question": "which?"}), say("done")])
    child = Agent(child_llm, "helper", "Helps.", input_type=EchoIn)
    parent_llm = CapturingLlm([call("helper", {"text": "go"}), say("done")])
    parent = make_agent(parent_llm).tool(child).tool(HUMAN)
    human = ScriptedHuman(["unused"])
    assert await parent.run({}, human=human) == Answer("done")
    assert human.asked == []
    assert all(spec.name != "ask_user" for spec in child_llm.advertised[0])
    assert any(spec.name == "ask_user" for spec in parent_llm.advertised[0])


def test_duplicate_tool_names_are_rejected_when_registering() -> None:
    agent = make_agent(ScriptedLlm([])).tool(echo)
    with pytest.raises(ValueError, match="unique"):
        agent.tool(echo)


@pytest.mark.parametrize("invalid_first", [False, True])
async def test_a_valid_submission_preempts_sibling_calls_in_the_same_step(
    invalid_first: bool,
) -> None:
    executed = False

    @tool(description="Records that it ran.")
    async def spy(input: EchoIn) -> str:
        nonlocal executed
        executed = True
        return "ran"

    calls = [tool_call("spy", {"text": "x"})]
    if invalid_first:
        calls.append(tool_call("final_answer", {"total": "invalid"}))
    calls.append(tool_call("final_answer", {"total": 1}))
    step = ModelStep(text="", tool_calls=tuple(calls))
    agent = (
        make_agent(ScriptedLlm([step])).output(SumOut, instructions="Report the total.").tool(spy)
    )
    result = await agent.run({})
    assert result == Answer(SumOut(total=1))
    assert executed is False


async def test_plain_text_from_a_typed_agent_stalls_then_the_nudged_submission_wins() -> None:
    llm = CapturingLlm([say("here is prose"), call("final_answer", {"total": 2})])
    agent = make_agent(llm).output(SumOut)
    result = await agent.run({})
    assert result == Answer(SumOut(total=2))
    # The nudge arrived as transcript input for the second step.
    nudged = llm.transcripts[1]
    assert any("final_answer" in getattr(entry, "text", "") for entry in nudged)


async def test_an_invalid_submission_returns_a_readable_error_and_the_model_retries() -> None:
    llm = CapturingLlm(
        [call("final_answer", {"total": "not a number"}), call("final_answer", {"total": 3})]
    )
    agent = make_agent(llm).output(SumOut)
    result = await agent.run({})
    assert result == Answer(SumOut(total=3))
    returns = [entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns)]
    assert returns and returns[0].returns[0].content.startswith("ERROR: invalid final_answer")


async def test_ask_user_ends_the_turn_with_an_ask() -> None:
    step = call(
        "ask_user",
        {
            "kind": "approval",
            "question": "Create the PI draft?",
            "payload": {"action": "create_pi", "total": 4000},
        },
    )
    agent = make_agent(ScriptedLlm([step])).tool(HUMAN)
    result, events = await run_collecting(agent, {})
    assert isinstance(result, Ask)
    assert result.kind == "approval"
    assert result.question == "Create the PI draft?"
    assert result.payload == {"action": "create_pi", "total": 4000}
    issued = [event for event in events if isinstance(event, AskIssued)]
    assert len(issued) == 1
    assert issued[0].ask_id == result.ask_id


async def test_a_valid_ask_preempts_sibling_side_effects() -> None:
    executed = False

    @tool(description="Records that it ran.")
    async def spy(input: EchoIn) -> str:
        nonlocal executed
        executed = True
        return "ran"

    step = ModelStep(
        text="",
        tool_calls=(
            tool_call("spy", {"text": "x"}),
            tool_call("ask_user", {"question": "which one?"}),
        ),
    )
    agent = make_agent(ScriptedLlm([step])).tool(HUMAN).tool(spy)
    result = await agent.run({})
    assert isinstance(result, Ask)
    assert executed is False


async def test_a_final_answer_outranks_an_ask_in_the_same_step() -> None:
    step = ModelStep(
        text="",
        tool_calls=(
            tool_call("ask_user", {"question": "sure?"}),
            tool_call("final_answer", {"total": 9}),
        ),
    )
    agent = make_agent(ScriptedLlm([step])).output(SumOut).tool(HUMAN)
    result = await agent.run({})
    assert result == Answer(SumOut(total=9))


async def test_an_invalid_ask_is_rejected_readably_and_the_model_recovers() -> None:
    llm = CapturingLlm([call("ask_user", {"wrong": True}), say("fine, no question")])
    agent = make_agent(llm).tool(HUMAN)
    result = await agent.run({})
    assert result == Answer("fine, no question")
    returns = [entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns)]
    assert returns and returns[0].returns[0].content.startswith("ERROR: invalid ask_user")


async def test_ask_user_is_unknown_without_a_registered_human() -> None:
    llm = CapturingLlm([call("ask_user", {"question": "hm?"}), say("done")])
    agent = make_agent(llm)
    result = await agent.run({})
    assert result == Answer("done")
    returns = [entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns)]
    assert returns and "unknown tool: ask_user" in returns[0].returns[0].content
    assert all(spec.name != "ask_user" for spec in llm.advertised[0])


async def test_reserved_names_cannot_be_registered() -> None:
    @tool(name="final_answer", description="An impostor.")
    async def impostor(input: EchoIn) -> str:
        return "no"

    with pytest.raises(ValueError, match="reserved"):
        make_agent(ScriptedLlm([])).tool(impostor)


async def test_unknown_tool_calls_stall_until_exhausted() -> None:
    steps = [call("nope", {}) for _ in range(3)]
    agent = make_agent(ScriptedLlm(steps))
    with pytest.raises(Exhausted, match="no progress"):
        await agent.run({})


async def test_the_step_limit_ends_the_run_with_exhausted() -> None:
    steps = [call("echo", {"text": "x"}) for _ in range(4)]
    agent = make_agent(ScriptedLlm(steps)).tool(echo).with_max_steps(4)
    with pytest.raises(Exhausted, match="step limit"):
        await agent.run({})


async def test_same_step_calls_run_concurrently() -> None:
    barrier = asyncio.Barrier(2)

    @tool(description="Waits for its sibling.")
    async def left(input: Empty) -> str:
        async with asyncio.timeout(2):
            await barrier.wait()
        return "left"

    @tool(name="right", description="Waits for its sibling.")
    async def right_tool(input: Empty) -> str:
        async with asyncio.timeout(2):
            await barrier.wait()
        return "right"

    step = ModelStep(text="", tool_calls=(tool_call("left", {}), tool_call("right", {})))
    agent = make_agent(ScriptedLlm([step, say("both ran")])).tool(left).tool(right_tool)
    result = await agent.run({})
    assert result == Answer("both ran")


async def test_step_events_count_the_loop() -> None:
    agent = make_agent(ScriptedLlm([call("echo", {"text": "x"}), say("done")])).tool(echo)
    _, events = await run_collecting(agent, {})
    steps = [event.step for event in events if isinstance(event, StepStart)]
    assert steps == [1, 2]


def costing(step: ModelStep, input: int, output: int) -> ModelStep:
    return replace(step, usage=Usage(input=input, output=output))


async def test_each_steps_usage_is_reported_before_its_calls_run() -> None:
    script = [costing(call("echo", {"text": "x"}), 100, 5), costing(say("done"), 130, 3)]
    agent = make_agent(ScriptedLlm(script)).tool(echo)
    _, events = await run_collecting(agent, {})
    kinds = [type(event).__name__ for event in events]
    assert kinds.index("UsageReported") < kinds.index("ToolInputStart")
    reported = [event.usage for event in events if isinstance(event, UsageReported)]
    assert reported == [Usage(input=100, output=5), Usage(input=130, output=3)]


async def test_a_step_that_says_nothing_of_its_cost_reports_nothing() -> None:
    agent = make_agent(ScriptedLlm([say("done")]))
    _, events = await run_collecting(agent, {})
    assert not any(isinstance(event, UsageReported) for event in events)


async def test_a_sub_agents_usage_reaches_the_root_stream() -> None:
    sub = Agent(
        ScriptedLlm([costing(say("sub says hi"), 40, 4)]), "helper", "helps", input_type=EchoIn
    )
    script = [
        costing(call("helper", {"text": "hello"}), 100, 5),
        costing(say("root answer"), 150, 6),
    ]
    agent = make_agent(ScriptedLlm(script)).tool(sub)
    _, events = await run_collecting(agent, {})
    reported = [event.usage for event in events if isinstance(event, UsageReported)]
    assert reported == [
        Usage(input=100, output=5),
        Usage(input=40, output=4),
        Usage(input=150, output=6),
    ]


async def test_a_registered_sub_agents_answer_lands_in_the_parent_transcript() -> None:
    sub = Agent(ScriptedLlm([say("sub says hi")]), "helper", "helps", input_type=EchoIn)
    llm = CapturingLlm([call("helper", {"text": "hello"}), say("done")])
    agent = make_agent(llm).tool(sub)
    result = await agent.run({})
    assert result == Answer("done")
    returns = [entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns)]
    assert returns and returns[0].returns[0].content == "sub says hi"


async def test_an_empty_text_step_is_nudged_to_speak() -> None:
    llm = CapturingLlm([say(""), say("actual words")])
    agent = make_agent(llm)
    result = await agent.run({})
    assert result == Answer("actual words")


@pytest.mark.internals
async def test_internal_tool_failure_stays_private() -> None:
    from void_agent.core.agent.rules import INTERNAL_TOOL_ERROR

    @tool(description="Blows up.")
    async def broken(input: EchoIn) -> str:
        raise RuntimeError("secret detail")

    agent = make_agent(ScriptedLlm([call("broken", {"text": "x"})])).tool(broken)
    events, queue = EventSender.channel(64)
    with pytest.raises(Internal):
        await agent.run({}, events)
    collected: list[AgentEvent] = []
    while not queue.empty():
        collected.append(queue.get_nowait())
    errors = [event for event in collected if isinstance(event, ToolOutputError)]
    assert errors and errors[0].error_text == INTERNAL_TOOL_ERROR
    assert all("secret detail" not in event.error_text for event in errors)


async def test_a_rejected_tool_error_is_model_readable() -> None:
    @tool(description="Always refuses.")
    async def refuser(input: EchoIn) -> str:
        raise Rejected("out of stock")

    llm = CapturingLlm([call("refuser", {"text": "x"}), say("routed around")])
    agent = make_agent(llm).tool(refuser)
    result = await agent.run({})
    assert result == Answer("routed around")
    returns = [entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns)]
    assert returns and returns[0].returns[0].content == "ERROR: out of stock"


async def test_cancellation_is_never_swallowed_by_a_crashing_sibling() -> None:
    @tool(description="Crashes first.")
    async def crasher(input: Empty) -> str:
        raise RuntimeError("boom")

    @tool(description="Gets cancelled.")
    async def cancelled(input: Empty) -> str:
        raise asyncio.CancelledError()

    step = ModelStep(text="", tool_calls=(tool_call("crasher", {}), tool_call("cancelled", {})))
    agent = make_agent(ScriptedLlm([step])).tool(crasher).tool(cancelled)
    with pytest.raises(asyncio.CancelledError):
        await agent.run({})


async def test_a_non_object_output_type_is_a_construction_error() -> None:
    with pytest.raises(TypeError, match="object-shaped"):
        make_agent(ScriptedLlm([])).output(str)


@pytest.mark.internals
async def test_an_empty_step_from_a_typed_agent_gets_a_truthful_nudge() -> None:
    from void_agent.core.agent.rules import MUST_SUBMIT_EMPTY_RULE

    llm = CapturingLlm([say(""), call("final_answer", {"total": 7})])
    agent = make_agent(llm).output(SumOut)
    result = await agent.run({})
    assert result == Answer(SumOut(total=7))
    nudged = llm.transcripts[1]
    assert any(getattr(entry, "text", "") == MUST_SUBMIT_EMPTY_RULE for entry in nudged)


async def test_a_sub_agents_voice_and_steps_stay_out_of_the_root_stream_by_default() -> None:
    sub = Agent(ScriptedLlm([say("sub says hi")]), "helper", "helps", input_type=EchoIn)
    agent = make_agent(ScriptedLlm([call("helper", {"text": "hello"}), say("root answer")]))
    agent.tool(sub)
    _, events = await run_collecting(agent, {})
    deltas = "".join(event.delta for event in events if isinstance(event, TextDelta))
    assert "sub says hi" not in deltas
    steps = [event.step for event in events if isinstance(event, StepStart)]
    assert steps == [1, 2]  # the sub-agent's own step rhythm never leaks


# ── the human attends the run ─────────────────────────────────────────────


def names(events: list[AgentEvent]) -> list[str]:
    return [type(event).__name__ for event in events]


async def test_an_attended_ask_is_answered_in_place_and_the_step_goes_on() -> None:
    llm = CapturingLlm(
        [
            ModelStep(
                text="",
                tool_calls=(
                    tool_call(
                        "ask_user",
                        {"kind": "choice", "question": "color?", "options": ["red", "blue"]},
                    ),
                    tool_call("echo", {"text": "sibling"}),
                ),
            ),
            say("blue it is"),
        ]
    )
    agent = make_agent(llm).tool(echo).tool(HUMAN)
    human = ScriptedHuman(["blue"])
    result, events = await run_collecting(agent, {}, human)
    assert result == Answer("blue it is")
    # The answer is the ask_user call's own result; the sibling ran after it.
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert [r.name for r in returns.returns] == ["ask_user", "echo"]
    assert returns.returns[0].content == '{"answer":"blue"}'
    assert [n for n in names(events) if n.startswith("Ask")] == ["AskIssued", "AskAnswered"]
    assert human.asked[0].question == "color?" and human.asked[0].options == ("red", "blue")


async def test_a_dropped_ask_ends_the_turn_before_any_sibling_and_marks_the_card() -> None:
    ran: list[str] = []

    @tool(description="Records that it ran.")
    async def side_effect(input: EchoIn) -> str:
        ran.append(input.text)
        return "done"

    agent = (
        make_agent(
            ScriptedLlm(
                [
                    ModelStep(
                        text="",
                        tool_calls=(
                            tool_call("ask_user", {"question": "sure?"}),
                            tool_call("side_effect", {"text": "boom"}),
                        ),
                    )
                ]
            )
        )
        .tool(side_effect)
        .tool(HUMAN)
    )
    result, events = await run_collecting(agent, {}, ScriptedHuman([]))
    assert isinstance(result, Ask) and result.question == "sure?"
    assert ran == []
    assert [n for n in names(events) if n.startswith("Ask")] == ["AskIssued", "AskDropped"]


async def test_a_sub_agents_question_reaches_the_attendant_directly() -> None:
    """No parent relays anything: the sub-agent asks, the person answers,
    the sub-agent continues and hands its answer up like any tool."""
    sub = Agent(
        ScriptedLlm([call("ask_user", {"question": "ship 60?"}), say("shipped 60")]),
        "trader",
        "trades",
        input_type=EchoIn,
    ).tool(HUMAN)
    llm = CapturingLlm([call("trader", {"text": "go"}), say("relayed nothing")])
    root = make_agent(llm).tool(sub)
    human = ScriptedHuman(["yes"])
    result, events = await run_collecting(root, {}, human)
    assert result == Answer("relayed nothing")
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content == "shipped 60"
    assert [ask.question for ask in human.asked] == ["ship 60?"]
    # The sub-agent's card passes through the root's stream: someone can answer it.
    assert "AskIssued" in names(events) and "AskAnswered" in names(events)


async def test_an_unanswered_question_deep_in_the_tree_ends_every_run_above() -> None:
    sub = Agent(
        ScriptedLlm([call("ask_user", {"question": "ship 60?"})]),
        "trader",
        "trades",
        input_type=EchoIn,
    ).tool(HUMAN)
    root = make_agent(ScriptedLlm([call("trader", {"text": "go"}), say("never")])).tool(sub)
    result, events = await run_collecting(root, {}, ScriptedHuman([]))
    assert isinstance(result, Ask) and result.question == "ship 60?"
    outputs = [e for e in events if isinstance(e, ToolOutputAvailable)]
    assert outputs[-1].output == HELD_TOOL_OUTPUT


class DraftIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: float


def make_draft(ran: list[DraftIn]) -> Tool:
    @tool(
        description="Creates a draft; large ones are signed by a human.",
        approval=lambda draft: f"total {draft.total} exceeds 100" if draft.total > 100 else None,
    )
    async def create_draft(input: DraftIn) -> dict[str, str]:
        ran.append(input)
        return {"draft_id": "D-1"}

    return create_draft


async def test_a_gated_tool_runs_right_after_the_click() -> None:
    ran: list[DraftIn] = []
    llm = CapturingLlm([call("create_draft", {"total": 270}), say("D-1 is ready")])
    agent = make_agent(llm).tool(make_draft(ran))
    human = ScriptedHuman([True])
    result, events = await run_collecting(agent, {}, human)
    assert result == Answer("D-1 is ready")
    assert ran == [DraftIn(total=270)]
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content == '{"draft_id":"D-1"}'
    assert human.asked[0].call == Call(tool="create_draft", input={"total": 270.0})
    assert names(events)[1:6] == [
        "ToolInputStart",
        "ToolInputAvailable",
        "AskIssued",
        "AskAnswered",
        "ToolOutputAvailable",
    ]


async def test_a_declined_gate_reads_as_a_rejection_the_model_can_route_around() -> None:
    ran: list[DraftIn] = []
    llm = CapturingLlm([call("create_draft", {"total": 270}), say("understood, no draft")])
    agent = make_agent(llm).tool(make_draft(ran))
    result, events = await run_collecting(agent, {}, ScriptedHuman([False]))
    assert result == Answer("understood, no draft") and ran == []
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content.startswith("ERROR: the user declined create_draft")
    assert any(isinstance(e, ToolOutputError) for e in events)


async def test_an_unanswered_gate_ends_the_turn_with_the_call_on_the_card() -> None:
    ran: list[DraftIn] = []
    agent = make_agent(ScriptedLlm([call("create_draft", {"total": 270})])).tool(make_draft(ran))
    result, events = await run_collecting(agent, {})  # nobody attends
    assert isinstance(result, Ask) and result.kind == "approval"
    assert result.call == Call(tool="create_draft", input={"total": 270.0})
    assert ran == []
    assert names(events)[1:5] == [
        "ToolInputStart",
        "ToolInputAvailable",
        "AskIssued",
        "ToolOutputAvailable",
    ]


async def test_a_gate_inside_a_workflow_is_answered_without_any_relay() -> None:
    ran: list[DraftIn] = []
    create_draft = make_draft(ran)  # never registered — only the workflow knows it

    @tool(description="Prepares and creates a draft in one go.")
    async def draft_flow(input: DraftIn, events: EventSender) -> dict[str, Any]:
        draft = await create_draft.invoke({"total": input.total}, events)
        return {"flow": "done", "draft": draft}

    llm = CapturingLlm([call("draft_flow", {"total": 270}), say("done")])
    agent = make_agent(llm).tool(draft_flow)
    result, _ = await run_collecting(agent, {}, ScriptedHuman([True]))
    assert result == Answer("done") and ran == [DraftIn(total=270)]
    returns = next(entry for entry in llm.transcripts[1] if isinstance(entry, ToolReturns))
    assert returns.returns[0].content == '{"flow":"done","draft":{"draft_id":"D-1"}}'

    dropped, _ = await run_collecting(
        make_agent(ScriptedLlm([call("draft_flow", {"total": 270})])).tool(draft_flow),
        {},
        ScriptedHuman([]),
    )
    assert (
        isinstance(dropped, Ask)
        and dropped.call is not None
        and dropped.call.tool == "create_draft"
    )


async def test_a_sub_agent_inherits_the_attendant_of_the_enclosing_run() -> None:
    ran: list[DraftIn] = []
    sub = Agent(
        ScriptedLlm([call("create_draft", {"total": 270}), say("drafted")]),
        "trader",
        "trades",
        input_type=EchoIn,
    ).tool(make_draft(ran))
    root = make_agent(ScriptedLlm([call("trader", {"text": "go"}), say("done")])).tool(sub)
    human = ScriptedHuman([True])
    result, _ = await run_collecting(root, {}, human)
    assert result == Answer("done") and ran == [DraftIn(total=270)]
    assert human.asked[0].call is not None and human.asked[0].call.tool == "create_draft"


def test_an_agent_lists_what_it_registered_in_order() -> None:
    @tool(description="one")
    async def first(input: dict[str, int]) -> int:
        return 1

    @tool(description="two")
    async def second(input: dict[str, int]) -> int:
        return 2

    agent = Agent(ScriptedLlm([]), "a", "b").tool(first).tool(second).tool(HUMAN)
    assert agent.tool_names == ("first", "second")  # ask_user is the runtime's, not a tool
