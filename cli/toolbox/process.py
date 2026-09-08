"""Running a command in the root. The arguments are a list, never a string
a shell expands — which is what makes the signature card readable: what
the person sees is exactly the argv that runs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from cli.toolbox.root import ToolboxError, clip, resolve

RUN_TIMEOUT_SECONDS = 120.0


async def run_in(
    root: Path,
    command: str,
    args: Sequence[str] | None = None,
    cwd: str | None = None,
    timeout: float = RUN_TIMEOUT_SECONDS,
) -> str:
    """Run a command in the root and report how it went.

    Arguments are a list, never a string a shell expands — no pipe, no
    redirect, no `&&`. That is not a permission (whether this call may run
    at all is the application's to decide, in `/mcp`); it is what makes the
    signature card readable, because what the person sees is exactly the
    argv that runs."""
    where = resolve(root, cwd)
    if not where.is_dir():
        raise ToolboxError(f"{cwd} is not a directory")
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *(args or ()),
            cwd=where,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        raise ToolboxError(f"{command}: command not found") from None
    except OSError as error:
        raise ToolboxError(f"cannot run {command}: {error}") from error
    try:
        out, _ = await asyncio.wait_for(process.communicate(), max(0.1, timeout))
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ToolboxError(f"{command} took longer than {timeout:.0f}s and was stopped") from None
    said = clip(out.decode("utf-8", "replace").rstrip())
    return f"exit {process.returncode}\n{said}" if said else f"exit {process.returncode}"
