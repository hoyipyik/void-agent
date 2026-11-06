"""`python -m void_agent.evals`: the commands.

    list   CONFIG [choice]          the cases a run would ask
    mount  CONFIG --persona NAME    what each server offers that persona
    run    CONFIG [choice] [...]    ask them, and write every record down
    report RUN                      read a run back into RUN/report.md
    show   RUN KEY                  one case, step by step

A choice is any of `--case GLOB`, `--persona NAME`, `--tag TAG`, each
repeatable. Keys come from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`), and so does whatever `${NAME}` a server's address
or headers name.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from void_agent.evals.case import Setup
from void_agent.evals.load import load_builder
from void_agent.evals.model import make_llm
from void_agent.evals.mount import MountError, mounted
from void_agent.evals.record import Record
from void_agent.evals.report import (
    calls_of,
    counted,
    missed,
    render,
    render_case,
    steps_of,
    summarize,
)
from void_agent.evals.run import run_cases
from void_agent.evals.store import RunDir
from void_agent.evals.suite import (
    PERSON_ROLE,
    Case,
    Limits,
    ModelSpec,
    Suite,
    SuiteError,
    environment,
    load_suite,
    select,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m void_agent.evals",
        description="Ask a suite's questions and record every run.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def chooses(command: argparse.ArgumentParser) -> None:
        command.add_argument("config", type=Path, help="the suite's eval.toml")
        command.add_argument("--case", action="append", default=[], help="a case id or glob")
        command.add_argument("--persona", action="append", default=[])
        command.add_argument("--tag", action="append", default=[])

    chooses(commands.add_parser("list", help="the cases a run would ask"))
    mount = commands.add_parser("mount", help="what each server offers a persona")
    mount.add_argument("config", type=Path)
    mount.add_argument("--persona", required=True)
    run = commands.add_parser("run", help="ask the cases and write every record down")
    chooses(run)
    run.add_argument("--jobs", type=int, default=2, help="cases asked at once (2)")
    run.add_argument("--repeat", type=int, default=1, help="times each case is asked (1)")
    run.add_argument("--label", default="", help="a word for the run's folder name")
    run.add_argument("--out", type=Path, help="the folder runs go in (the suite's `runs`)")
    run.add_argument("--resume", type=Path, help="a run to finish: its missing cases are asked")
    run.add_argument("--model", help="provider:model, in place of the suite's")
    run.add_argument("--effort", help="the effort, in place of the suite's")
    run.add_argument(
        "--turn-seconds", type=float, help="one turn's limit, in place of the suite's"
    )
    report = commands.add_parser("report", help="read a run back into RUN/report.md")
    report.add_argument("run", type=Path)
    show = commands.add_parser("show", help="one case of a run, step by step")
    show.add_argument("run", type=Path)
    show.add_argument("key")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    out: TextIO = sys.stdout,
) -> int:
    arguments = _parser().parse_args(argv)
    env = environment() if env is None else env
    try:
        match arguments.command:
            case "list":
                return _list(arguments, out)
            case "mount":
                return asyncio.run(_mount(arguments, env, out))
            case "run":
                return _run(arguments, env, out)
            case "report":
                return _report(arguments, out)
            case _:
                return _show(arguments, out)
    except (SuiteError, MountError, FileNotFoundError) as error:
        print(f"evals: {error}", file=out)
        return 2


def _chosen(arguments: argparse.Namespace) -> tuple[Suite, tuple[Case, ...]]:
    suite = load_suite(arguments.config)
    cases = select(suite.cases, ids=arguments.case, personas=arguments.persona, tags=arguments.tag)
    return suite, cases


def _list(arguments: argparse.Namespace, out: TextIO) -> int:
    suite, cases = _chosen(arguments)
    for case in cases:
        tags = f" [{', '.join(case.tags)}]" if case.tags else ""
        print(f"{case.id:<36} {case.persona:<8} {case.turns[0].text}{tags}", file=out)
    print(f"{len(cases)} cases of {len(suite.cases)} in {suite.name}", file=out)
    return 0


async def _mount(arguments: argparse.Namespace, env: Mapping[str, str], out: TextIO) -> int:
    suite = load_suite(arguments.config)
    persona = suite.personas.get(arguments.persona)
    if persona is None:
        raise SuiteError(f"the suite has no persona `{arguments.persona}`")
    async with mounted(persona, env) as mount:
        for server, published in mount.catalog.items():
            print(f"{server}: instructions {published['instructions_fingerprint']}", file=out)
            for tool in published["tools"]:
                state = "offered" if tool["name"] in mount.offered else "withheld"
                print(f"  {tool['name']:<28} {state:<9} {tool['fingerprint']}", file=out)
        print(f"{len(mount.offered)} offered, {len(mount.withheld)} withheld", file=out)
    return 0


def _model(suite: Suite, arguments: argparse.Namespace) -> ModelSpec:
    model = suite.model
    if arguments.model:
        provider, colon, name = arguments.model.partition(":")
        if not colon or not name:
            raise SuiteError("--model is provider:model, such as openai:gpt-6-luna")
        model = replace(model, provider=provider, model=name)
    if arguments.effort is not None:
        model = replace(model, effort=arguments.effort)
    return model


def _run(arguments: argparse.Namespace, env: Mapping[str, str], out: TextIO) -> int:
    suite, cases = _chosen(arguments)
    if not cases:
        print("evals: no case is chosen", file=out)
        return 1
    suite = replace(suite, model=_model(suite, arguments))
    if arguments.turn_seconds is not None:
        suite = replace(suite, limits=Limits(turn_seconds=arguments.turn_seconds))
    llm, close = make_llm(suite.model, env)
    person, close_person = make_llm(suite.person.model, env) if suite.person else (None, None)
    build = load_builder(suite.agent, suite.paths)
    run = (
        RunDir.open(arguments.resume)
        if arguments.resume
        else RunDir.create(arguments.out or suite.runs, arguments.label)
    )
    setup = Setup(
        build=build,
        llm=llm,
        env=env,
        context=suite.context,
        limits=suite.limits,
        person=person,
        person_rounds=suite.person.rounds if suite.person else 0,
        person_role=suite.person.role if suite.person else PERSON_ROLE,
    )
    print(f"{len(cases)} cases, {arguments.repeat} times each -> {run.path}", file=out)

    def landed(done: int, total: int, record: Record) -> None:
        line = (
            f"[{done:>3}/{total}] {record['key']:<36} {record['outcome']:<9}"
            f" {counted(steps_of(record), 'step'):>8} {counted(len(calls_of(record)), 'call'):>8}"
            f" {record.get('seconds', 0):>5.0f} s"
        )
        found = missed(record)
        print(line + (f"  ✗ {'; '.join(found)}" if found else ""), file=out, flush=True)

    async def asked() -> None:
        try:
            await run_cases(
                suite,
                cases,
                setup,
                run,
                jobs=arguments.jobs,
                repeat=arguments.repeat,
                label=arguments.label,
                on_record=landed,
            )
        finally:
            await close()
            if close_person is not None:
                await close_person()

    try:
        asyncio.run(asked())
    except KeyboardInterrupt:
        print(
            f"stopped: resume with --resume {run.path}",
            file=out,
        )
        return 130
    report = run.path / "report.md"
    report.write_text(render(summarize(run)), encoding="utf-8")
    print(f"report: {report}", file=out)
    return 0


def _report(arguments: argparse.Namespace, out: TextIO) -> int:
    run = RunDir.open(arguments.run)
    report = run.path / "report.md"
    report.write_text(render(summarize(run)), encoding="utf-8")
    print(f"report: {report}", file=out)
    return 0


def _show(arguments: argparse.Namespace, out: TextIO) -> int:
    run = RunDir.open(arguments.run)
    if arguments.key not in run.written():
        raise SuiteError(f"{run.path} has no case `{arguments.key}`")
    print("\n".join(render_case(run.record(arguments.key), detail=True)), file=out)
    return 0
