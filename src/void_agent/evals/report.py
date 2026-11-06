"""A run read back, as Markdown: what was run; per persona how its cases
ended and what they cost; per tool how often it was called, how each
call ended, how long it ran, how long its results were and how many a
later turn would read cut short, and which arguments the model filled
in; the cases that missed what they expected; the cases that need what
the agent cannot do yet; then each case — its question, its calls and
its answer, as they were.

Nothing is judged here that a run cannot check by itself: whether an
answer is right is for whoever reads it, with the case's intent beside it.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from void_agent.evals.record import Record
from void_agent.evals.store import RunDir

OUTCOME_ORDER = ("answered", "asked", "failed", "cancelled", "error")
# How much of a call's input or result one line of the report shows.
INPUT_CHARS = 300
OUTPUT_CHARS = 1500


@dataclass
class ToolTally:
    calls: int = 0
    keys: set[str] = field(default_factory=lambda: cast("set[str]", set()))
    statuses: Counter[str] = field(default_factory=lambda: Counter[str]())
    seconds: list[float] = field(default_factory=lambda: cast("list[float]", []))
    lengths: list[int] = field(default_factory=lambda: cast("list[int]", []))
    cut: int = 0
    arguments: Counter[str] = field(default_factory=lambda: Counter[str]())

    @property
    def cases(self) -> int:
        return len(self.keys)

    @property
    def longest(self) -> int:
        return max(self.lengths, default=0)


@dataclass
class PersonaTally:
    cases: int = 0
    outcomes: Counter[str] = field(default_factory=lambda: Counter[str]())
    steps: int = 0
    calls: int = 0
    seconds: float = 0.0
    input: int = 0
    output: int = 0


@dataclass
class Summary:
    name: str
    manifest: Record
    records: list[Record]
    personas: dict[str, PersonaTally]
    tools: dict[str, ToolTally]
    missed: dict[str, list[str]]
    limit: int | None


def _fields(value: Any) -> dict[str, Any]:
    """A record's table, or an empty one where it has none."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def calls_of(record: Record) -> list[Record]:
    return [call for turn in record["turns"] for call in turn["calls"]]


def steps_of(record: Record) -> int:
    return sum(len(turn["steps"]) for turn in record["turns"])


def missed(record: Record) -> list[str]:
    """What a case expected that its run did not do."""
    expect: dict[str, Any] = record["case"]["expect"]
    called = Counter(call["name"] for call in calls_of(record))
    found: list[str] = []
    wanted = expect.get("outcome")
    if wanted and wanted != record["outcome"]:
        found.append(f"ended {record['outcome']}, expected {wanted}")
    found += [f"never called {name}" for name in expect.get("uses", []) if not called[name]]
    any_of: list[str] = expect.get("uses_any", [])
    if any_of and not any(called[name] for name in any_of):
        found.append(f"called none of {', '.join(any_of)}")
    found += [
        f"called {name}, which it should avoid"
        for name in expect.get("avoids", [])
        if called[name]
    ]
    most = expect.get("max_calls")
    free: list[str] = expect.get("uncounted", [])
    made = sum(n for name, n in called.items() if name not in free)
    if most is not None and made > most:
        besides = f" besides {', '.join(free)}" if free else ""
        found.append(f"made {counted(made, 'call')}{besides}, expected at most {most}")
    return found


def summarize(run: RunDir) -> Summary:
    manifest = run.manifest()
    records = run.records()
    limit: int | None = _fields(manifest.get("context")).get("tool_output_limit")
    personas: dict[str, PersonaTally] = {}
    tools: dict[str, ToolTally] = {}
    misses: dict[str, list[str]] = {}
    for record in records:
        tally = personas.setdefault(record["persona"], PersonaTally())
        tally.cases += 1
        tally.outcomes[record["outcome"]] += 1
        tally.steps += steps_of(record)
        tally.calls += len(calls_of(record))
        tally.seconds += record.get("seconds", 0.0)
        usage = _fields(record.get("usage"))
        tally.input += usage.get("input", 0)
        tally.output += usage.get("output", 0)
        for call in calls_of(record):
            tool = tools.setdefault(call["name"], ToolTally())
            tool.calls += 1
            tool.keys.add(record["key"])
            tool.statuses[call.get("status", "unfinished")] += 1
            if "seconds" in call:
                tool.seconds.append(call["seconds"])
            if "chars" in call:
                tool.lengths.append(call["chars"])
                if limit and call["chars"] > limit:
                    tool.cut += 1
            if isinstance(call.get("input"), dict):
                tool.arguments.update(cast("dict[str, Any]", call["input"]).keys())
        found = missed(record)
        if found:
            misses[record["key"]] = found
    return Summary(
        name=manifest.get("suite", run.path.name),
        manifest=manifest,
        records=records,
        personas=personas,
        tools=dict(sorted(tools.items(), key=lambda item: -item[1].calls)),
        missed=misses,
        limit=limit,
    )


# ── Markdown ──────────────────────────────────────────────────────────────


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text)} chars)"


def _row(cells: Iterable[object]) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def _table(head: Sequence[str], rows: Iterable[Iterable[object]]) -> list[str]:
    return [_row(head), _row("---" for _ in head), *(_row(row) for row in rows)]


def _mean(values: Sequence[float]) -> str:
    return f"{statistics.fmean(values):.1f}" if values else "-"


def _fence(text: str) -> list[str]:
    """Text as it was, in a fence no Markdown inside it can close."""
    return ["~~~~text", text.rstrip(), "~~~~"]


def render_call(call: Record) -> str:
    line = f"`{call['name']}({_clip(_compact(call.get('input', {})), INPUT_CHARS)})`"
    status = call.get("status", "unfinished")
    seconds = f", {call['seconds']:.2f} s" if "seconds" in call else ""
    if status == "ok":
        line += f" → ok, {call.get('chars', 0)} chars{seconds}"
    elif status in ("refused", "error"):
        line += f" → {status}: {call.get('error', '')}{seconds}"
    elif status == "broken":
        line += f" → broken: {call.get('error', '')}{seconds}"
    else:
        line += f" → {status}{seconds}"
    return line


def render_case(record: Record, *, detail: bool = False) -> list[str]:
    case = record["case"]
    calls = calls_of(record)
    lines = [
        f"### {record['key']}",
        "",
        " · ".join(
            [
                record["persona"],
                f"**{record['outcome']}**",
                counted(steps_of(record), "step"),
                counted(len(calls), "call"),
                f"{record.get('seconds', 0):.1f} s",
                *([", ".join(case["tags"])] if case["tags"] else []),
            ]
        ),
        "",
    ]
    if case.get("intent"):
        lines += [f"*Intent:* {case['intent']}", ""]
    if case.get("gap"):
        lines += [f"*Gap:* {case['gap']}", ""]
    if record["key"] and missed(record):
        lines += [f"*Missed:* {'; '.join(missed(record))}", ""]
    if record.get("error"):
        error = record["error"]
        lines += [f"*Could not run ({error['stage']}):* {error['text']}", ""]
    for turn in record["turns"]:
        by = "*(person)* " if turn["user"].get("by") == "person" else ""
        lines += [f"> {by}{turn['user']['text']}", ""]
        if turn["user"]["attach"]:
            lines += [f"Attached: {', '.join(turn['user']['attach'])}", ""]
        lines += [
            f"*Asked in place:* {asked['question']} → {asked['answer'] or '(no answer)'}"
            for asked in turn.get("asks", [])
            if asked.get("answer") is not None or turn.get("outcome") != "asked"
        ]
        if detail:
            lines += _steps(turn)
        else:
            lines += [
                f"{index}. {render_call(call)}" for index, call in enumerate(turn["calls"], 1)
            ]
            if turn["calls"]:
                lines.append("")
        if turn.get("answer") is not None:
            lines += _fence(_compact(turn["answer"]))
        elif turn.get("ask"):
            lines += [f"*Asked:* {turn['ask']['question']}"]
        if turn.get("error"):
            lines += [f"*{turn['outcome']}:* {turn['error']['text']}"]
        lines.append("")
    return lines


def _steps(turn: Record) -> list[str]:
    """Each step of a turn: what the model said of its reasoning, what it
    wrote, the calls it made and what they returned."""
    lines: list[str] = []
    calls = {call["id"]: call for call in turn["calls"]}
    for step in turn["steps"]:
        usage = _fields(step.get("usage"))
        lines.append(
            f"**Step {step['step']}** · {step.get('seconds', 0):.1f} s"
            + (f" · {usage.get('input', 0)} in / {usage.get('output', 0)} out" if usage else "")
        )
        lines.append("")
        lines += _read(step.get("read", []))
        for thought in step.get("reasoning", []):
            lines += [f"*Reasoning:* {thought}", ""]
        if step.get("text") and step.get("calls"):
            lines += [step["text"], ""]
        for made in step.get("calls", []):
            call = calls.get(made["id"], {"name": made["name"], "input": made["args"]})
            lines += [f"- {render_call(call)}", ""]
            if call.get("status") == "ok":
                lines += [*_fence(_clip(_compact(call.get("output")), OUTPUT_CHARS)), ""]
        if step.get("error"):
            lines += [f"*Failed:* {step['error']['type']}: {step['error']['text']}", ""]
    return lines


def _read(entries: Sequence[Record]) -> list[str]:
    """What a step read that says something a reader of the case cannot
    see elsewhere: the earlier turns as the model was handed them, and the
    runtime's own nudges. The system prompt is named by its length; this
    turn's question and the tool results are shown where they belong."""
    lines: list[str] = []
    for entry in entries:
        if "system" in entry:
            lines += [f"*Read the system prompt* ({len(entry['system'])} chars)", ""]
        elif "assistant" in entry:
            lines += ["*Read an earlier answer:*", *_fence(entry["assistant"]), ""]
        elif "user_text" in entry:
            lines += [f"*Read the runtime's nudge:* {entry['user_text']}", ""]
    return lines


def render(summary: Summary) -> str:
    manifest = summary.manifest
    records = summary.records
    model = manifest.get("model", {})
    outcomes = Counter(record["outcome"] for record in records)
    lines = [
        f"# {summary.name}",
        "",
        f"- Agent: `{manifest.get('agent', '?')}`",
        f"- Model: {model.get('provider', '?')} `{model.get('model', '?')}`"
        + (f", effort {model['effort']}" if model.get("effort") else ""),
        f"- Started {manifest.get('started_at', '?')},"
        f" finished {manifest.get('finished_at', '-')}",
    ]
    for found in manifest.get("revisions", []):
        dirty = ", with changes" if found.get("dirty") else ""
        lines.append(f"- `{found['path']}` at {found['commit'][:10]} ({found['branch']}{dirty})")
    lines += [
        f"- {len(records)} cases: "
        + " · ".join(f"{outcomes[name]} {name}" for name in OUTCOME_ORDER if outcomes[name]),
        "",
        "## Personas",
        "",
        *_table(
            [
                "persona",
                "cases",
                *OUTCOME_ORDER,
                "steps/case",
                "calls/case",
                "s/case",
                "in",
                "out",
            ],
            (
                [
                    name,
                    tally.cases,
                    *(tally.outcomes[outcome] for outcome in OUTCOME_ORDER),
                    f"{tally.steps / tally.cases:.1f}",
                    f"{tally.calls / tally.cases:.1f}",
                    f"{tally.seconds / tally.cases:.0f}",
                    tally.input,
                    tally.output,
                ]
                for name, tally in summary.personas.items()
            ),
        ),
        "",
        "## Tools",
        "",
    ]
    if summary.limit:
        lines += [
            f"*Cut* counts the results longer than the {summary.limit} characters"
            " a later turn reads of one.",
            "",
        ]
    lines += _table(
        [
            "tool",
            "calls",
            "cases",
            "ok",
            "refused",
            "broken",
            "other",
            "s avg",
            "s max",
            "chars avg",
            "chars max",
            "cut",
        ],
        (
            [
                name,
                tally.calls,
                tally.cases,
                tally.statuses["ok"],
                tally.statuses["refused"],
                tally.statuses["broken"],
                tally.calls
                - tally.statuses["ok"]
                - tally.statuses["refused"]
                - tally.statuses["broken"],
                _mean(tally.seconds),
                f"{max(tally.seconds, default=0):.1f}",
                _mean([float(length) for length in tally.lengths]),
                tally.longest,
                tally.cut,
            ]
            for name, tally in summary.tools.items()
        ),
    )
    lines += ["", "### Arguments the model filled in", ""]
    lines += [
        f"- **{name}**: "
        + ", ".join(f"{key} x{count}" for key, count in tally.arguments.most_common())
        for name, tally in summary.tools.items()
        if tally.arguments
    ]
    by_key = {record["key"]: record for record in records}
    lines += ["", "## Expectations missed", ""]
    lines += [
        f"- **{key}** ({by_key[key]['persona']}): {'; '.join(found)}"
        for key, found in summary.missed.items()
    ] or ["None."]
    gaps = [record for record in records if record["case"].get("gap")]
    if gaps:
        lines += ["", "## Gaps", ""]
        lines += [
            f"- **{record['key']}** ({record['persona']}, {record['outcome']}):"
            f" {record['case']['gap']}"
            for record in gaps
        ]
    lines += ["", "## Cases", ""]
    for record in records:
        lines += render_case(record)
    return "\n".join(lines).rstrip() + "\n"
