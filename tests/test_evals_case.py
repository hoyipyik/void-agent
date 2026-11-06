"""One case run as its persona: each turn on a scripted model and a fake
server, written down whole — what the model read and returned each step,
every call with its whole result, every event, how each turn ended."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult
from tests.mcp_fakes import FakeMcp, descriptor, text_result

from void_agent import (
    HUMAN,
    INTERNAL_PUBLIC_TEXT,
    Agent,
    EventSender,
    Llm,
    ModelStep,
    ScriptedLlm,
    ScriptedStep,
    Tool,
    ToolSpec,
    TranscriptEntry,
    Usage,
    call,
    say,
)
from void_agent.evals.case import Setup, run_case
from void_agent.evals.suite import Case, Context, Expect, Limits, Persona, ServerSpec, Turn
from void_agent.mcp import McpServer

SYSTEM = "You answer questions about the shop."
SERVER = ServerSpec(name="shop", url="http://shop.example/mcp")


class Shop(FakeMcp):
    """A server whose tools each answer in a way of their own: a result,
    or a failure of the transport."""

    def __init__(self, answers: Mapping[str, CallToolResult | Exception]) -> None:
        super().__init__([descriptor(name, f"{name} in the shop") for name in answers])
        self._answers = answers

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, arguments))
        answer = self._answers[name]
        if isinstance(answer, Exception):
            raise answer
        return answer


SEARCH = {"search": text_result('{"customer":"CUST-7"}')}


def build(llm: Llm, tools: Sequence[Tool]) -> Agent:
    agent = Agent(llm, "shop", "answers about the shop").with_system(SYSTEM)
    agent = agent.prompt(lambda history: list(history)).tool(HUMAN)
    for capability in tools:
        agent.tool(capability)
    return agent


def setup(
    script: Sequence[ScriptedStep] | Llm,
    answers: Mapping[str, CallToolResult | Exception] | None = None,
    *,
    context: Context | None = None,
    limits: Limits | None = None,
    builder: Callable[[Llm, Sequence[Tool]], Agent] = build,
    env: dict[str, str] | None = None,
) -> Setup:
    def opener(spec: ServerSpec) -> McpServer:
        return McpServer(client=Shop(answers if answers is not None else SEARCH))

    return Setup(
        build=builder,
        llm=script if not isinstance(script, Sequence) else ScriptedLlm(list(script)),
        open=opener,
        env=env or {},
        context=context or Context(),
        limits=limits or Limits(),
    )


def case(*turns: str, replies: tuple[str | bool, ...] = ()) -> Case:
    return Case(
        id="who-is-northwind",
        persona="manager",
        turns=tuple(Turn(text) for text in turns),
        replies=replies,
        expect=Expect(uses=("search",)),
    )


MANAGER = Persona("manager", servers={"shop": SERVER})


async def run(the_case: Case, the_setup: Setup, persona: Persona = MANAGER) -> dict[str, Any]:
    record = await run_case(the_case, persona, the_setup)
    json.dumps(record)  # a record is written as JSON, whole
    return record


async def test_a_case_records_what_the_model_read_and_what_it_returned_each_step() -> None:
    record = await run(
        case("Who is Northwind?"),
        setup([call("search", {"q": "Northwind"}, call_id="c1"), say("Northwind is CUST-7.")]),
    )

    assert record["outcome"] == "answered"
    (turn,) = record["turns"]
    assert turn["answer"] == "Northwind is CUST-7."
    first, second = turn["steps"]
    system, user = first["read"]
    assert system["system"].startswith(SYSTEM)
    assert user == {"user": [{"text": "Who is Northwind?"}]}
    assert first["calls"] == [{"id": "c1", "name": "search", "args": {"q": "Northwind"}}]
    assert second["read"] == [
        {"assistant_step": 1},
        {"tool_returns": [{"id": "c1", "name": "search", "content": '{"customer":"CUST-7"}'}]},
    ]
    assert second["text"] == "Northwind is CUST-7."
    assert second["calls"] == []


async def test_the_tools_the_model_was_offered_are_recorded_as_it_saw_them() -> None:
    record = await run(case("Who is Northwind?"), setup([say("I do not know.")]))

    names = [spec["name"] for spec in record["specs"]]
    assert names == ["search", "ask_user"]
    assert record["specs"][0]["description"] == "search in the shop"
    assert record["offered"] == ["search"]
    assert record["catalog"]["shop"]["tools"][0]["fingerprint"].startswith("sha256:")


async def test_a_tool_call_is_recorded_with_its_whole_result_and_how_long_it_took() -> None:
    long = "x" * 5000
    record = await run(
        case("Who is Northwind?"),
        setup(
            [call("search", {"q": "Northwind"}, call_id="c1"), say("Done.")],
            {"search": text_result(long)},
        ),
    )

    (made,) = record["turns"][0]["calls"]
    assert made["id"] == "c1"
    assert made["name"] == "search"
    assert made["step"] == 1
    assert made["input"] == {"q": "Northwind"}
    assert made["status"] == "ok"
    assert made["output"] == long
    assert made["chars"] == 5000
    assert made["seconds"] >= 0
    assert made["source"] == "shop"


async def test_a_refused_call_is_recorded_in_the_servers_words() -> None:
    refused = text_result("no such customer", is_error=True)
    record = await run(
        case("Who is Northwind?"),
        setup(
            [call("search", {"q": "Northwind"}, call_id="c1"), say("Not found.")],
            {"search": refused},
        ),
    )

    (made,) = record["turns"][0]["calls"]
    assert made["status"] == "refused"
    assert made["error"] == "search: no such customer"
    returned = record["turns"][0]["steps"][1]["read"][1]["tool_returns"][0]
    assert returned["content"] == "ERROR: search: no such customer"


async def test_a_broken_call_fails_the_turn_and_keeps_the_detail_nobody_else_saw() -> None:
    cut = ConnectionResetError("the server cut the connection")
    record = await run(
        case("Who is Northwind?"),
        setup([call("search", {"q": "Northwind"}, call_id="c1"), say("Done.")], {"search": cut}),
    )

    assert record["outcome"] == "failed"
    (turn,) = record["turns"]
    (made,) = turn["calls"]
    assert made["status"] == "broken"
    assert "the server cut the connection" in made["detail"]
    assert turn["error"]["text"] == INTERNAL_PUBLIC_TEXT
    assert "the server cut the connection" in turn["error"]["detail"]
    assert turn["parts"][-1] == {"type": "data-error", "data": {"text": INTERNAL_PUBLIC_TEXT}}


async def test_a_persona_is_offered_only_the_tools_its_filters_leave() -> None:
    answers = {
        **SEARCH,
        "sales_total": text_result("{}"),
        "sales_prices": text_result("{}"),
    }
    clerk = Persona("clerk", servers={"shop": SERVER}, exclude=("sales_*",))

    record = await run(case("Who is Northwind?"), setup([say("I do not know.")], answers), clerk)

    assert record["offered"] == ["search"]
    assert record["withheld"] == ["sales_total", "sales_prices"]
    assert [spec["name"] for spec in record["specs"]] == ["search", "ask_user"]


async def test_a_later_turn_reads_an_earlier_tool_result_cut_at_the_context_limit() -> None:
    record = await run(
        case("Who is Northwind?", "And their code again?"),
        setup(
            [call("search", {"q": "Northwind"}, call_id="c1"), say("CUST-7."), say("CUST-7.")],
            {"search": text_result("y" * 500)},
            context=Context(tool_output_limit=100),
        ),
    )

    later = record["turns"][1]["steps"][0]["read"]
    assert later[1] == {"user": [{"text": "Who is Northwind?"}]}
    (remembered,) = later[2]["assistant"].splitlines()[:1]
    assert remembered == '[tool search({"q":"Northwind"}) → "' + "y" * 99 + "…]"
    assert later[3] == {"user": [{"text": "And their code again?"}]}


async def test_a_later_turn_reads_back_only_the_calls_of_the_tools_the_context_names() -> None:
    record = await run(
        case("Who is Northwind?", "And again?"),
        setup(
            [call("search", {"q": "Northwind"}), say("CUST-7."), say("CUST-7.")],
            context=Context(tools=()),
        ),
    )

    assert record["turns"][1]["steps"][0]["read"][2] == {"assistant": "CUST-7."}


async def test_a_question_to_the_person_ends_the_turn_and_the_next_turn_answers_it() -> None:
    record = await run(
        case("How much did Northwind buy?", "The German one."),
        setup(
            [call("ask_user", {"question": "Which Northwind?"}), say("Northwind GmbH bought 10.")]
        ),
    )

    assert record["outcome"] == "answered"
    asked, answered = record["turns"]
    assert asked["outcome"] == "asked"
    assert asked["ask"]["question"] == "Which Northwind?"
    assert answered["answer"] == "Northwind GmbH bought 10."
    assert "Which Northwind?" in answered["steps"][0]["read"][2]["assistant"]


async def test_a_scripted_reply_answers_the_models_question_in_place() -> None:
    record = await run(
        case("How much did Northwind buy?", replies=("The German one.",)),
        setup(
            [call("ask_user", {"question": "Which Northwind?"}, call_id="a1"), say("10 units.")]
        ),
    )

    (turn,) = record["turns"]
    assert turn["outcome"] == "answered"
    returned = turn["steps"][1]["read"][1]["tool_returns"][0]
    assert returned == {"id": "a1", "name": "ask_user", "content": '{"answer":"The German one."}'}
    assert "AskAnswered" in [event["type"] for event in turn["events"]]


async def test_a_turn_that_runs_out_of_steps_is_failed_with_the_reason() -> None:
    def limited(llm: Llm, tools: Sequence[Tool]) -> Agent:
        return build(llm, tools).with_max_steps(2)

    record = await run(
        case("Who is Northwind?"),
        setup(
            [call("search", {"q": "Northwind"}), call("search", {"q": "Northwind"})],
            builder=limited,
        ),
    )

    assert record["outcome"] == "failed"
    assert "step limit" in record["turns"][0]["error"]["text"]


class Stuck:
    async def step(
        self, transcript: Sequence[TranscriptEntry], tools: Sequence[ToolSpec], events: EventSender
    ) -> ModelStep:
        await asyncio.sleep(60)
        return say("too late")


async def test_a_turn_that_outlasts_its_time_is_cancelled_and_kept() -> None:
    record = await run(
        case("Who is Northwind?", "And then?"),
        setup(Stuck(), limits=Limits(turn_seconds=0.05)),
    )

    assert record["outcome"] == "cancelled"
    (turn,) = record["turns"]
    assert turn["error"]["text"] == "the turn took longer than 0.05 s"
    assert turn["parts"][-1] == {"type": "data-cancelled", "data": {}}
    assert turn["steps"][0]["error"]["type"] == "CancelledError"


async def test_a_server_that_cannot_be_mounted_fails_the_case_before_any_turn() -> None:
    class Refusing(FakeMcp):
        async def __aenter__(self) -> Refusing:
            raise ConnectionRefusedError("nobody listens")

    def opener(spec: ServerSpec) -> McpServer:
        return McpServer(client=Refusing([]))

    refused = replace(setup([say("never")]), open=opener)

    record = await run(case("Who is Northwind?"), refused)

    assert record["outcome"] == "error"
    assert record["error"]["stage"] == "mount"
    assert "shop" in record["error"]["text"]
    assert record["turns"] == []


async def test_a_token_that_is_not_set_fails_only_the_cases_that_need_it() -> None:
    secret = ServerSpec(name="shop", url="http://shop.example/mcp", headers={"A": "${TOKEN}"})
    persona = Persona("manager", servers={"shop": secret})

    record = await run(case("Who is Northwind?"), setup([say("never")]), persona)

    assert record["outcome"] == "error"
    assert "TOKEN" in record["error"]["text"]


async def test_the_events_of_a_turn_are_kept_in_order_with_their_times() -> None:
    record = await run(
        case("Who is Northwind?"),
        setup([call("search", {"q": "Northwind"}, call_id="c1"), say("Northwind is CUST-7.")]),
    )

    events = record["turns"][0]["events"]
    kinds = [event["type"] for event in events]
    assert kinds[:4] == [
        "StepStart",
        "ToolInputStart",
        "ToolInputAvailable",
        "ToolOutputAvailable",
    ]
    assert kinds.count("TextDelta") == 1
    texts = [event for event in events if event["type"] == "TextDelta"]
    assert texts[0]["delta"] == "Northwind is CUST-7."
    assert all(later["at"] >= earlier["at"] for earlier, later in pairwise(events))


async def test_what_each_round_trip_cost_is_added_up_for_the_case() -> None:
    record = await run(
        case("Who is Northwind?"),
        setup(
            [
                ModelStep(text="", tool_calls=call("search", {}).tool_calls, usage=Usage(10, 5)),
                ModelStep(text="CUST-7.", usage=Usage(30, 7, cache_read=8)),
            ]
        ),
    )

    assert record["usage"] == {
        "input": 40,
        "output": 12,
        "cache_read": 8,
        "cache_write": 0,
        "round_trips": 2,
    }
    assert record["turns"][0]["steps"][1]["usage"] == {
        "input": 30,
        "output": 7,
        "cache_read": 8,
        "cache_write": 0,
    }


async def test_a_later_turn_reads_back_the_calls_of_the_mounted_tools_and_the_last_plan() -> None:
    def planning(llm: Llm, tools: Sequence[Tool]) -> Agent:
        return build(llm, tools).with_plan()

    plan = {"items": [{"id": "1", "title": "find Northwind", "status": "completed"}]}
    record = await run(
        case("Who is Northwind?", "And again?"),
        setup(
            [
                ModelStep(
                    text="",
                    tool_calls=(
                        call("update_plan", plan).tool_calls[0],
                        call("search", {"q": "Northwind"}).tool_calls[0],
                    ),
                ),
                say("CUST-7."),
                say("CUST-7."),
            ],
            context=Context(tools="mounted"),
            builder=planning,
        ),
    )

    remembered = record["turns"][1]["steps"][0]["read"][2]["assistant"]
    assert "[tool search(" in remembered
    assert "update_plan" not in remembered
    assert "[plan: ✓ find Northwind]" in remembered


async def test_each_turn_says_when_it_started_in_the_cases_own_time() -> None:
    record = await run(
        case("Who is Northwind?", "And again?"),
        setup([say("CUST-7."), say("CUST-7.")]),
    )

    first, second = record["turns"]
    assert 0 <= first["at"] <= second["at"] <= record["seconds"]


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000100e221bc330000000049454e44ae426082"
)


async def test_an_attached_file_reaches_the_model_beside_the_question(tmp_path: Path) -> None:
    (tmp_path / "order.png").write_bytes(PNG)
    attached = Case(
        "order", "manager", (Turn("What is on this order?", (tmp_path / "order.png",)),)
    )

    record = await run(attached, setup([say("An order.")]))

    assert record["turns"][0]["steps"][0]["read"][1] == {
        "user": [{"text": "What is on this order?"}, {"image": "image/png", "bytes": len(PNG)}]
    }
    assert record["turns"][0]["user"]["attach"] == ["order.png"]


async def test_an_attachment_that_cannot_be_read_fails_the_case_naming_it(tmp_path: Path) -> None:
    missing = Case("order", "manager", (Turn("What is on this order?", (tmp_path / "gone.pdf",)),))

    record = await run(missing, setup([say("never")]))

    assert record["outcome"] == "error"
    assert record["error"]["stage"] == "attach"
    assert "gone.pdf" in record["error"]["text"]


async def test_an_attached_text_file_reaches_the_model_as_its_text(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Ship the panels on Friday.", encoding="utf-8")
    attached = Case(
        "notes", "manager", (Turn("What do the notes say?", (tmp_path / "notes.txt",)),)
    )

    record = await run(attached, setup([say("Friday.")]))

    read = "".join(part["text"] for part in record["turns"][0]["steps"][0]["read"][1]["user"])
    assert "What do the notes say?" in read
    assert "Ship the panels on Friday." in read
    assert record["turns"][0]["user"]["attach"] == ["notes.txt"]


async def test_a_file_that_is_neither_image_pdf_nor_text_fails_the_case_naming_it(
    tmp_path: Path,
) -> None:
    (tmp_path / "parts.bin").write_bytes(b"\x00\x01\x02")
    attached = Case("parts", "manager", (Turn("What is this?", (tmp_path / "parts.bin",)),))

    record = await run(attached, setup([say("never")]))

    assert record["outcome"] == "error"
    assert record["error"]["stage"] == "attach"
    assert "parts.bin" in record["error"]["text"]


async def test_an_agent_that_cannot_be_built_fails_the_case_before_its_turn() -> None:
    def broken(llm: Llm, tools: Sequence[Tool]) -> Agent:
        raise RuntimeError("no prompt")

    record = await run(case("Who is Northwind?"), setup([say("never")], builder=broken))

    assert record["outcome"] == "error"
    assert record["error"]["stage"] == "build"
    assert "no prompt" in record["error"]["detail"]


async def test_a_command_server_that_does_not_answer_its_mount_in_time_fails_the_case() -> None:
    """The suite's `timeout` bounds a server run as a command as it does one
    reached by URL: a process that never speaks the protocol does not hold
    the case forever."""
    silent = ServerSpec(
        name="files",
        command=sys.executable,
        args=("-c", "import sys; sys.stdin.read()"),
        timeout=0.5,
    )
    the_setup = Setup(build=build, llm=ScriptedLlm([say("never")]))

    record = await run_case(
        case("Who is Northwind?"), Persona("reader", servers={"files": silent}), the_setup
    )

    assert record["outcome"] == "error"
    assert record["error"]["stage"] == "mount"
    assert "in time" in record["error"]["text"]
