"""The evals from the command line: list what a suite would ask, refuse a
run it cannot make in words that say why, and read a run back."""

from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from tests.mcp_fakes import FakeMcp, descriptor, text_result
from tests.test_evals_case import build
from tests.test_evals_suite import suite_in

from void_agent import ScriptedLlm, call, say
from void_agent.evals.case import Setup
from void_agent.evals.main import main
from void_agent.evals.run import run_cases
from void_agent.evals.store import RunDir
from void_agent.evals.suite import load_suite
from void_agent.mcp import McpServer


def cli(*argv: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), env=env or {}, out=out)
    return code, out.getvalue()


def test_list_names_the_cases_a_run_would_ask(tmp_path: Path) -> None:
    code, text = cli("list", str(suite_in(tmp_path)), "--persona", "manager")

    assert code == 0
    assert "manager-top-customers" in text
    assert "Who bought the most this year?" in text
    assert "clerk-status" not in text
    assert "2 cases" in text


def test_a_run_whose_model_has_no_key_says_which_variable(tmp_path: Path) -> None:
    (tmp_path / "agents" / "agents").mkdir(parents=True)
    (tmp_path / "agents" / "agents" / "orders.py").write_text(
        "def build(llm, tools):\n    raise AssertionError('never built')\n"
    )

    code, text = cli("run", str(suite_in(tmp_path)))

    assert code == 2
    assert "OPENAI_API_KEY" in text


def test_a_run_that_chooses_no_case_says_so(tmp_path: Path) -> None:
    code, text = cli("run", str(suite_in(tmp_path)), "--case", "nothing-*")

    assert code == 1
    assert "no case" in text


async def written_run(tmp_path: Path) -> RunDir:
    suite = load_suite(suite_in(tmp_path))
    run = RunDir.create(suite.runs)

    def opener(spec: object) -> McpServer:
        return McpServer(client=FakeMcp([descriptor("search")], text_result("found")))

    setup = Setup(
        build=build,
        llm=ScriptedLlm([call("search", {"q": "Northwind"}), say("Northwind is CUST-7.")]),
        open=opener,
        env={"TOKEN_MANAGER": "token-manager"},
    )
    await run_cases(suite, suite.cases[:1], setup, run)
    return run


async def test_report_writes_the_markdown_into_the_run(tmp_path: Path) -> None:
    run = await written_run(tmp_path)

    code, text = cli("report", str(run.path))

    assert code == 0
    report = (run.path / "report.md").read_text()
    assert "### manager-top-customers" in report
    assert str(run.path / "report.md") in text


async def test_show_prints_one_case_step_by_step_with_what_the_tools_returned(
    tmp_path: Path,
) -> None:
    run = await written_run(tmp_path)

    code, text = cli("show", str(run.path), "manager-top-customers")

    assert code == 0
    assert "**Step 1**" in text
    assert '`search({"q":"Northwind"})` → ok, 5 chars' in text
    assert "found" in text
    assert "Northwind is CUST-7." in text


async def test_show_prints_an_earlier_answer_as_a_later_turn_was_handed_it(
    tmp_path: Path,
) -> None:
    suite = load_suite(suite_in(tmp_path))
    run = RunDir.create(suite.runs)
    setup = Setup(
        build=build,
        llm=ScriptedLlm([say("Northwind bought 10."), say("CUST-7.")]),
        open=lambda spec: McpServer(client=FakeMcp([descriptor("search")])),
        env={"TOKEN_MANAGER": "token-manager"},
    )
    two_turns = replace(suite.cases[1], turns=suite.cases[1].turns[:1] * 2)
    await run_cases(suite, [two_turns], setup, run)

    code, text = cli("show", str(run.path), "manager-follow-up")

    assert code == 0
    assert "*Read an earlier answer:*" in text
    assert text.count("Northwind bought 10.") == 2


def test_a_run_whose_person_has_no_key_says_which_variable(tmp_path: Path) -> None:
    from tests.test_evals_suite import CONFIG

    config = CONFIG + '\n[person]\nprovider = "anthropic"\nmodel = "claude-haiku-5-5"\n'
    path = suite_in(tmp_path, **{"eval.toml": config})

    code, text = cli("run", str(path), env={"OPENAI_API_KEY": "sk-test"})

    assert code == 2
    assert "ANTHROPIC_API_KEY" in text


def test_the_installed_package_runs_as_a_command(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "void_agent.evals", "list", str(suite_in(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "3 cases of 3" in done.stdout


def test_a_caller_reads_and_chooses_a_suite_from_the_package_itself(tmp_path: Path) -> None:
    import void_agent.evals as evals

    suite = evals.load_suite(suite_in(tmp_path))
    chosen = evals.select(suite.cases, personas=["clerk"])

    assert [case.id for case in chosen] == ["clerk-status"]
