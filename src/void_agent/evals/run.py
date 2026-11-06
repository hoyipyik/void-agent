"""A run of the chosen cases: asked side by side, up to `jobs` at a time,
each `repeat` times; every record written as it lands and announced.
The manifest says what was run — the suite, the agent, the model, the
void-agent that asked, which revision of each repository the agent came
from — before the first case starts, and when the last one ended."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

from void_agent.evals.case import Setup, run_case
from void_agent.evals.record import Record, jsonable
from void_agent.evals.store import RunDir
from void_agent.evals.suite import Case, Suite

OnRecord = Callable[[int, int, Record], None]


def key_of(case: Case, repeat: int, repeats: int) -> str:
    return case.id if repeats == 1 else f"{case.id}~{repeat}"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def revision(path: Path) -> Record | None:
    """The commit a folder's repository is at, and whether it has changes
    beyond it — so a record says which prompt and which tools ran."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "path": str(path),
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime() -> Record | None:
    """The void-agent that asks the cases: the commit a git install came
    from, or, installed from a folder, that checkout's revision. Never the
    package's own folder, which inside a project's virtualenv is the
    project's repository."""
    try:
        installed = metadata.distribution("void-agent")
    except metadata.PackageNotFoundError:
        return None
    found: Record = json.loads(installed.read_text("direct_url.json") or "{}")
    url: str = found.get("url", "")
    vcs: Record = found.get("vcs_info") or {}
    commit: str | None = vcs.get("commit_id")
    if commit:
        return {"version": installed.version, "url": url, "commit": commit}
    if url.startswith("file://"):
        checkout = revision(Path(unquote(urlparse(url).path)))
        if checkout is not None:
            return {"version": installed.version, **checkout}
    return {"version": installed.version, "url": url}


def manifest_of(suite: Suite, keys: Sequence[str], label: str = "") -> Record:
    return {
        "suite": suite.name,
        "config": str(suite.config),
        "label": label,
        "agent": suite.agent,
        "model": jsonable(asdict(suite.model)),
        "context": jsonable(asdict(suite.context)),
        "limits": jsonable(asdict(suite.limits)),
        "person": jsonable(asdict(suite.person)) if suite.person else None,
        "personas": {
            name: {
                "description": persona.description,
                "only": list(persona.only),
                "exclude": list(persona.exclude),
                "servers": sorted(persona.servers),
            }
            for name, persona in suite.personas.items()
        },
        "void_agent": runtime(),
        "revisions": [found for path in suite.paths if (found := revision(path)) is not None],
        "keys": list(keys),
        "started_at": _now(),
    }


async def run_cases(
    suite: Suite,
    cases: Sequence[Case],
    setup: Setup,
    run: RunDir,
    *,
    jobs: int = 2,
    repeat: int = 1,
    label: str = "",
    on_record: OnRecord | None = None,
) -> list[Record]:
    """Ask every case `repeat` times that `run` has no record of yet, and
    return the records this call wrote."""
    work = [(case, n) for case in cases for n in range(1, repeat + 1)]
    keys = [key_of(case, n, repeat) for case, n in work]
    manifest = run.manifest()
    if manifest:
        manifest["keys"] = list(dict.fromkeys([*manifest.get("keys", []), *keys]))
        manifest.setdefault("resumed_at", []).append(_now())
    else:
        manifest = manifest_of(suite, keys, label)
    run.write_manifest(manifest)

    done = run.written()
    todo = [(case, n) for case, n in work if key_of(case, n, repeat) not in done]
    gate = asyncio.Semaphore(max(jobs, 1))
    written: list[Record] = []

    async def ask(case: Case, n: int) -> None:
        async with gate:
            record = await run_case(
                case, suite.personas[case.persona], setup, key=key_of(case, n, repeat)
            )
            record["repeat"] = n
            run.write(record)
            written.append(record)
            if on_record is not None:
                on_record(len(written), len(todo), record)

    await asyncio.gather(*(ask(case, n) for case, n in todo))
    manifest["finished_at"] = _now()
    run.write_manifest(manifest)
    order = {key: index for index, key in enumerate(keys)}
    return sorted(written, key=lambda record: order[record["key"]])
