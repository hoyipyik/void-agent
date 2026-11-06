"""Evals: a set of questions put to an agent, every run recorded whole.

A suite is two kinds of file. The config (`eval.toml`) names the agent —
a builder that takes the model and the mounted tools, the way a server
builds one per turn — the model it runs on, the MCP servers its tools come
from, and the personas a question is asked as, each with its own
credentials and the tools it may be offered. The cases files hold the
questions: one conversation each, what it is for, and what it expects.

`python -m void_agent.evals run` asks each question as its persona and
writes down what happened: every step the model took, what it read and
what it returned, every tool call with its whole result, every event.
`report` reads a run back; nothing is judged while it runs.

The command is the usual way in; what it is built from is here for a
caller that drives a run itself. The files a suite is, the commands and a
run's folder are in README.md beside this file; it needs the `evals` extra
and the extra of each provider a suite names.
"""

from void_agent.evals.case import Setup, run_case
from void_agent.evals.main import main
from void_agent.evals.record import Record
from void_agent.evals.report import render, render_case, summarize
from void_agent.evals.run import run_cases
from void_agent.evals.store import RunDir
from void_agent.evals.suite import (
    Case,
    Expect,
    Persona,
    Suite,
    SuiteError,
    Turn,
    load_suite,
    select,
)

__all__ = [
    "Case",
    "Expect",
    "Persona",
    "Record",
    "RunDir",
    "Setup",
    "Suite",
    "SuiteError",
    "Turn",
    "load_suite",
    "main",
    "render",
    "render_case",
    "run_case",
    "run_cases",
    "select",
    "summarize",
]
