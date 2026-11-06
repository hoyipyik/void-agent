"""A run read back: per persona and per tool what happened, which cases
missed what they expected, and each case's question, calls and answer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mcp.types import CallToolResult
from tests.mcp_fakes import text_result
from tests.test_evals_case import Shop, build

from void_agent import ScriptedLlm, ScriptedStep, call, say
from void_agent.evals.case import Setup
from void_agent.evals.report import render, summarize
from void_agent.evals.run import run_cases
from void_agent.evals.store import RunDir
from void_agent.evals.suite import (
    Case,
    Context,
    Expect,
    ModelSpec,
    Persona,
    ServerSpec,
    Suite,
    Turn,
)
from void_agent.mcp import McpServer

SERVER = ServerSpec(name="shop", url="http://shop.example/mcp")
ANSWERS: dict[str, CallToolResult | Exception] = {
    "search": text_result("s" * 300),
    "order_list": text_result("no such customer", is_error=True),
    "sales_total": text_result("{}"),
}


def suite(tmp_path: Path, cases: Sequence[Case]) -> Suite:
    return Suite(
        name="shop",
        config=tmp_path / "eval.toml",
        agent="tests.test_evals_case:build",
        paths=(),
        model=ModelSpec("openai", "gpt-6-luna", "max"),
        personas={
            "manager": Persona("manager", servers={"shop": SERVER}),
            "clerk": Persona("clerk", servers={"shop": SERVER}),
        },
        cases=tuple(cases),
        runs=tmp_path / "runs",
        context=Context(tool_output_limit=200),
    )


async def ran(tmp_path: Path, script: Sequence[ScriptedStep], cases: Sequence[Case]) -> RunDir:
    def opener(spec: ServerSpec) -> McpServer:
        return McpServer(client=Shop(ANSWERS))

    the_suite = suite(tmp_path, cases)
    run = RunDir.create(the_suite.runs)
    setup = Setup(build=build, llm=ScriptedLlm(list(script)), open=opener)
    await run_cases(the_suite, the_suite.cases, setup, run, jobs=1)
    return run


TOP = Case(
    "top-customers",
    "manager",
    (Turn("Who bought the most?"),),
    intent="Ranks customers by invoiced sales.",
    expect=Expect(outcome="answered", uses=("sales_total",), avoids=("order_list",)),
)
STATUS = Case(
    "order-status",
    "clerk",
    (Turn("Where is ORD-2026-0042?"),),
    gap="no tool says where a shipment is",
    expect=Expect(uses_any=("search", "document_get"), max_calls=0),
)
SCRIPT: list[ScriptedStep] = [
    call("search", {"q": "Northwind", "limit": 5}),
    call("order_list", {"type": "invoice", "q": "Northwind"}),
    say("Northwind bought the most."),
    call("search", {"q": "ORD-2026-0042"}),
    say("It is finished."),
]


async def test_the_report_counts_each_tools_calls_and_how_they_ended(tmp_path: Path) -> None:
    summary = summarize(await ran(tmp_path, SCRIPT, [TOP, STATUS]))

    search = summary.tools["search"]
    assert (search.calls, search.cases) == (2, 2)
    assert search.statuses == {"ok": 2}
    assert search.arguments == {"q": 2, "limit": 1}
    assert summary.tools["order_list"].statuses == {"refused": 1}


async def test_a_result_longer_than_the_context_limit_is_counted_as_cut(tmp_path: Path) -> None:
    summary = summarize(await ran(tmp_path, SCRIPT, [TOP, STATUS]))

    assert summary.tools["search"].longest == 300
    assert summary.tools["search"].cut == 2


async def test_each_persona_is_tallied_by_how_its_cases_ended(tmp_path: Path) -> None:
    summary = summarize(await ran(tmp_path, SCRIPT, [TOP, STATUS]))

    assert summary.personas["manager"].outcomes == {"answered": 1}
    assert summary.personas["manager"].calls == 2
    assert summary.personas["clerk"].steps == 2


async def test_a_case_that_missed_what_it_expected_is_listed_with_what_it_missed(
    tmp_path: Path,
) -> None:
    summary = summarize(await ran(tmp_path, SCRIPT, [TOP, STATUS]))

    assert summary.missed == {
        "top-customers": [
            "never called sales_total",
            "called order_list, which it should avoid",
        ],
        "order-status": ["made 1 call, expected at most 0"],
    }


async def test_a_call_to_a_tool_the_suite_does_not_count_is_not_held_against_max_calls(
    tmp_path: Path,
) -> None:
    free = Case(
        "order-status",
        "clerk",
        (Turn("Where is ORD-2026-0042?"),),
        expect=Expect(max_calls=0, uncounted=("search",)),
    )
    script = [call("search", {"q": "ORD-2026-0042"}), say("It is finished.")]

    assert summarize(await ran(tmp_path, script, [free])).missed == {}


async def test_the_report_shows_each_case_with_its_question_calls_and_answer(
    tmp_path: Path,
) -> None:
    text = render(summarize(await ran(tmp_path, SCRIPT, [TOP, STATUS])))

    assert "## top-customers" in text
    assert "> Who bought the most?" in text
    assert 'search({"q":"Northwind","limit":5})' in text
    assert "order_list" in text and "refused" in text
    assert "Northwind bought the most." in text
    assert "no tool says where a shipment is" in text
    assert "| search | 2 | 2 |" in text


async def test_the_report_marks_what_the_person_said_and_what_they_answered_in_place(
    tmp_path: Path,
) -> None:
    from tests.test_evals_case import MANAGER, setup
    from tests.test_evals_person import Listening, briefed, says, with_person

    from void_agent.evals.case import run_case
    from void_agent.evals.report import render_case

    person = Listening([says("the british one"), says("ok"), says(None)])
    the_setup = with_person(
        setup(
            [
                say("Which Contoso?"),
                call("ask_user", {"question": "This year or last?"}),
                say("Contoso Ltd bought 10 this year."),
            ]
        ),
        person,
    )
    record = await run_case(briefed("how much did contoso buy"), MANAGER, the_setup)

    text = "\n".join(render_case(record))
    assert "> *(person)* the british one" in text
    assert "*Asked in place:* This year or last? → ok" in text
