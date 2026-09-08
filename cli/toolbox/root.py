"""The boundary. The root came from the command line; every path the
model offers is checked against it here, and nothing else in the toolbox
touches a path it did not get from `resolve` or `inside`.

Also what every answer goes through on its way back: an answer lands in
the model's context window, so `clip` caps its lines and its length — an
uncapped result can cost more than the file."""

from __future__ import annotations

from pathlib import Path

MAX_LINE = 240
MAX_CHARS = 40_000


class ToolboxError(Exception):
    """Something the model should read and can act on."""


def resolve(root: Path, path: str | None) -> Path:
    """A path under the root, or an error. The root came from the command
    line; `path` came from the model, so it is the one that is checked."""
    target = (root / (path or ".")).resolve()
    if not target.is_relative_to(root):
        raise ToolboxError(f"{path} is outside {root.name}; searches stay inside it")
    if not target.exists():
        raise ToolboxError(f"{path} does not exist in {root.name}")
    return target


def inside(root: Path, path: str) -> Path:
    """A path under the root, whether or not it exists yet — for the tools
    that create something. The root came from the command line; `path` came
    from the model, so it is the one that is checked."""
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ToolboxError(f"{path} is outside {root.name}; this server stays inside it")
    return target


def clip(text: str) -> str:
    """Long lines and long results, cut where a reader would cut them."""
    lines = [
        line if len(line) <= MAX_LINE else f"{line[:MAX_LINE]}… ({len(line)} chars)"
        for line in text.splitlines()
    ]
    out = "\n".join(lines)
    if len(out) > MAX_CHARS:
        out = f"{out[:MAX_CHARS]}\n… truncated at {MAX_CHARS} characters"
    return out
