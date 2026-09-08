"""Searching with ripgrep: the contents, the counts, the files it would
look at. One process per call, its arguments a list — never a shell
string, so nothing the model writes can become a command."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from cli.toolbox.root import ToolboxError, clip, resolve

# A grep answers into the model's context window, so it is capped in three
# directions: how many lines come back, how long a line may be, and how
# much text in total (`root.clip`). An uncapped search can cost more than
# the file.
MAX_RESULTS = 60
TIMEOUT_SECONDS = 20.0
# The name the release carries, which is what rides along in the bundle.
RG = "rg.exe" if sys.platform == "win32" else "rg"


def ripgrep() -> str | None:
    """Where `rg` is. The packed binary carries its own beside the bundle,
    which is not on anybody's PATH; everywhere else it is looked up. The
    carried one keeps the name it was released under, so on Windows it is
    `rg.exe` (`scripts/pack.py` puts it there)."""
    if getattr(sys, "frozen", False):
        carried = Path(getattr(sys, "_MEIPASS", "")) / RG
        if carried.is_file():
            return str(carried)
    return shutil.which("rg")


async def _run(root: Path, arguments: list[str]) -> str:
    """One ripgrep, run as a process with its arguments as a list — never a
    shell string, so nothing the model writes can become a command."""
    binary = ripgrep()
    if binary is None:
        raise ToolboxError("ripgrep (rg) is not on this machine; reading and listing still work")
    process = await asyncio.create_subprocess_exec(
        binary,
        *arguments,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        raw_out, raw_err = await asyncio.wait_for(process.communicate(), TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        raise ToolboxError(f"the search took longer than {TIMEOUT_SECONDS:.0f}s") from None
    out = raw_out.decode("utf-8", "replace")
    # rg exits 1 for "no matches", which is an answer, not a failure.
    if process.returncode not in (0, 1):
        raise ToolboxError(raw_err.decode("utf-8", "replace").strip() or "the search failed")
    return out


def flags(
    *,
    glob: str | None,
    ignore_case: bool,
    fixed: bool,
    hidden: bool,
) -> list[str]:
    built = ["--color", "never"]
    if glob:
        built += ["--glob", glob]
    if ignore_case:
        built.append("--ignore-case")
    if fixed:
        built.append("--fixed-strings")
    if hidden:
        built.append("--hidden")
    return built


async def search_in(
    root: Path,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool = False,
    fixed: bool = False,
    hidden: bool = False,
    context: int = 0,
    max_results: int = MAX_RESULTS,
) -> str:
    """The searching itself, apart from the server, so it can be tested."""
    target = resolve(root, path)
    limit = max(1, min(max_results, MAX_RESULTS))
    arguments = [
        "--line-number",
        "--no-heading",
        "--max-count",
        str(limit),
        *flags(glob=glob, ignore_case=ignore_case, fixed=fixed, hidden=hidden),
    ]
    if context:
        arguments += ["--context", str(max(0, min(context, 10)))]
    # `-e` and `--` so a pattern that starts with a dash stays a pattern.
    arguments += ["-e", pattern, "--", str(target)]
    found = await _run(root, arguments)
    if not found.strip():
        return f"no matches for {pattern!r}"
    root_prefix = f"{root}/"
    return clip(found.replace(root_prefix, ""))


async def count_in(
    root: Path,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    ignore_case: bool = False,
    fixed: bool = False,
) -> str:
    target = resolve(root, path)
    arguments = [
        "--count-matches",
        *flags(glob=glob, ignore_case=ignore_case, fixed=fixed, hidden=False),
        "-e",
        pattern,
        "--",
        str(target),
    ]
    found = await _run(root, arguments)
    if not found.strip():
        return f"no matches for {pattern!r}"
    return clip(found.replace(f"{root}/", ""))


async def files_in(root: Path, glob: str | None = None, path: str | None = None) -> str:
    target = resolve(root, path)
    arguments = ["--files", *flags(glob=glob, ignore_case=False, fixed=False, hidden=False)]
    arguments += ["--", str(target)]
    found = await _run(root, arguments)
    if not found.strip():
        return "no files match"
    return clip(found.replace(f"{root}/", ""))
