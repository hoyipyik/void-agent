"""A run: the chosen cases asked side by side, each record written to the
run's folder as it lands, so a run that stops is resumed where it was."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.mcp_fakes import FakeMcp, descriptor, text_result

from void_agent import (
    Agent,
    EventSender,
    Llm,
    ModelStep,
    Tool,
    ToolSpec,
    TranscriptEntry,
    say,
)
from void_agent.evals.case import Setup
from void_agent.evals.load import load_builder
from void_agent.evals.model import make_llm, request_extra
from void_agent.evals.run import run_cases
from void_agent.evals.store import RunDir
from void_agent.evals.suite import (
    Case,
    ModelSpec,
    Persona,
    ServerSpec,
    Suite,
    SuiteError,
    Turn,
)
from void_agent.mcp import McpServer

SERVER = ServerSpec(name="shop", url="http://shop.example/mcp")


class Answering:
    """A model that answers every question at once, counting how many
    questions it is in the middle of."""

    def __init__(self, hold: asyncio.Event | None = None) -> None:
        self.busy = 0
        self.most = 0
        self._hold = hold

    async def step(
        self, transcript: Sequence[TranscriptEntry], tools: Sequence[ToolSpec], events: EventSender
    ) -> ModelStep:
        self.busy += 1
        self.most = max(self.most, self.busy)
        if self._hold is not None:
            await self._hold.wait()
        await asyncio.sleep(0)
        self.busy -= 1
        return say("answered")


def build(llm: Llm, tools: Sequence[Tool]) -> Agent:
    agent = Agent(llm, "shop", "answers about the shop").prompt(lambda history: list(history))
    for capability in tools:
        agent.tool(capability)
    return agent


def suite(tmp_path: Path, *ids: str) -> Suite:
    return Suite(
        name="shop",
        config=tmp_path / "eval.toml",
        agent="tests.test_evals_run:build",
        paths=(),
        model=ModelSpec("openai", "gpt-6-luna", "max"),
        personas={"manager": Persona("manager", servers={"shop": SERVER})},
        cases=tuple(Case(id, "manager", (Turn(f"question {id}"),)) for id in ids),
        runs=tmp_path / "runs",
    )


def setup(llm: Llm) -> Setup:
    def opener(spec: ServerSpec) -> McpServer:
        return McpServer(client=FakeMcp([descriptor("search")], text_result("{}")))

    return Setup(build=build, llm=llm, open=opener)


async def test_a_run_writes_one_record_per_case_and_a_manifest(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one", "two")
    run = RunDir.create(the_suite.runs, "first")

    await run_cases(the_suite, the_suite.cases, setup(Answering()), run)

    assert run.path.parent == tmp_path / "runs"
    assert run.path.name.endswith("-first")
    manifest = json.loads((run.path / "run.json").read_text())
    assert manifest["suite"] == "shop"
    assert manifest["keys"] == ["one", "two"]
    assert manifest["model"] == {
        "provider": "openai",
        "model": "gpt-6-luna",
        "effort": "max",
        "extra": {},
    }
    assert manifest["finished_at"]
    assert [record["key"] for record in run.records()] == ["one", "two"]
    assert {record["outcome"] for record in run.records()} == {"answered"}


async def test_what_many_cases_saw_alike_is_written_once(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one", "two")
    run = RunDir.create(the_suite.runs)

    await run_cases(the_suite, the_suite.cases, setup(Answering()), run)

    one, two = (
        json.loads((run.path / "cases" / f"{key}.json").read_text()) for key in ("one", "two")
    )
    assert one["specs"] == two["specs"]
    assert one["catalog"] == two["catalog"]
    assert one["specs"].startswith("sha256:")
    assert [spec["name"] for spec in run.blob(one["specs"])] == ["search"]
    assert run.blob(one["catalog"])["shop"]["tools"][0]["name"] == "search"
    assert len(list((run.path / "blobs").iterdir())) == 2


async def test_a_resumed_run_asks_only_the_cases_it_has_no_record_of(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one", "two")
    run = RunDir.create(the_suite.runs)
    await run_cases(the_suite, the_suite.cases[:1], setup(Answering()), run)
    model = Answering()

    asked = await run_cases(the_suite, the_suite.cases, setup(model), RunDir.open(run.path))

    assert [record["key"] for record in asked] == ["two"]
    assert [record["key"] for record in run.records()] == ["one", "two"]


async def test_each_repeat_of_a_case_is_a_record_of_its_own(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one")
    run = RunDir.create(the_suite.runs)

    await run_cases(the_suite, the_suite.cases, setup(Answering()), run, repeat=3)

    assert [record["key"] for record in run.records()] == ["one~1", "one~2", "one~3"]
    assert {record["id"] for record in run.records()} == {"one"}


async def test_cases_are_asked_side_by_side_up_to_the_jobs_allowed(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one", "two", "three", "four")
    hold = asyncio.Event()
    model = Answering(hold)
    run = RunDir.create(the_suite.runs)

    running = asyncio.create_task(run_cases(the_suite, the_suite.cases, setup(model), run, jobs=2))
    while model.busy < 2:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    hold.set()
    await running

    assert model.most == 2


async def test_each_record_is_announced_as_it_lands(tmp_path: Path) -> None:
    the_suite = suite(tmp_path, "one", "two")
    landed: list[tuple[int, int, str]] = []

    def announce(done: int, total: int, record: dict[str, Any]) -> None:
        landed.append((done, total, record["key"]))

    await run_cases(
        the_suite,
        the_suite.cases,
        setup(Answering()),
        RunDir.create(the_suite.runs),
        on_record=announce,
    )

    assert sorted(landed) == [(1, 2, "one"), (2, 2, "two")] or sorted(landed) == [
        (1, 2, "two"),
        (2, 2, "one"),
    ]


def test_an_agent_is_found_by_its_module_or_its_file(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "shop_agent.py").write_text("def build(llm, tools):\n    return 'built'\n")
    (tmp_path / "loose.py").write_text("def make(llm, tools):\n    return 'made'\n")

    by_module = load_builder("pkg.shop_agent:build", (tmp_path,))
    by_file = load_builder(f"{tmp_path / 'loose.py'}:make", ())

    assert by_module(Answering(), []) == "built"  # type: ignore[comparison-overlap]
    assert by_file(Answering(), []) == "made"  # type: ignore[comparison-overlap]


def test_an_agent_that_cannot_be_found_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match=r"nowhere\.agent"):
        load_builder("nowhere.agent:build", (tmp_path,))
    with pytest.raises(SuiteError, match="module:attribute"):
        load_builder("just_a_name", ())


def test_a_model_whose_key_is_not_set_is_refused_naming_the_variable() -> None:
    with pytest.raises(SuiteError, match="OPENAI_API_KEY"):
        make_llm(ModelSpec("openai", "gpt-6-luna", "max"), {})


def test_the_effort_and_the_extra_go_into_every_request() -> None:
    spec = ModelSpec("openai", "gpt-6-luna", "max", {"reasoning": {"summary": "auto"}, "top_p": 1})

    assert request_extra(spec) == {"reasoning": {"effort": "max", "summary": "auto"}, "top_p": 1}
    assert request_extra(ModelSpec("openai", "gpt-6-luna")) == {}


def test_an_agent_file_may_declare_dataclasses(tmp_path: Path) -> None:
    (tmp_path / "typed_agent.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Settings:\n"
        "    name: str = 'shop'\n\n"
        "def build(llm, tools):\n"
        "    return Settings().name\n"
    )

    builder = load_builder(f"{tmp_path / 'typed_agent.py'}:build", ())

    assert builder(Answering(), []) == "shop"  # type: ignore[comparison-overlap]
