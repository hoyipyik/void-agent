"""The files under the root: read one or several, list a directory, draw
its tree, write, edit in place, move. Each takes the root and the model's
path and goes through `resolve` or `inside` before touching anything."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cli.toolbox.root import ToolboxError, clip, inside, resolve

MAX_BYTES = 400_000


def read_text(root: Path, path: str, head: int = 0, tail: int = 0) -> str:
    """A file's text, clipped. `head`/`tail` take the first or last N lines
    — cheaper than the whole file when the whole file is not the question."""
    target = resolve(root, path)
    if target.is_dir():
        raise ToolboxError(f"{path} is a directory; list_directory shows what is in it")
    if not head and not tail and target.stat().st_size > MAX_BYTES:
        head = 400
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise ToolboxError(f"cannot read {path}: {error}") from error
    lines = text.splitlines()
    if head:
        lines = lines[: max(1, head)]
    elif tail:
        lines = lines[-max(1, tail) :]
    return clip("\n".join(lines))


def list_dir(root: Path, path: str | None = None) -> str:
    """What is in a directory: names, a slash on the directories, a size on
    the files."""
    target = resolve(root, path)
    if not target.is_dir():
        raise ToolboxError(f"{path} is a file; read_file shows what is in it")
    rows = [
        f"{child.name}/" if child.is_dir() else f"{child.name}  {child.stat().st_size} bytes"
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    ]
    return clip("\n".join(rows)) if rows else "the directory is empty"


def read_many(root: Path, paths: Sequence[str]) -> str:
    """Several files in one answer. One that cannot be read says so in its
    own place and does not stop the rest."""
    if not paths:
        raise ToolboxError("name at least one file")
    blocks: list[str] = []
    for path in paths[:20]:
        try:
            blocks.append(f"===== {path}\n{read_text(root, path)}")
        except ToolboxError as error:
            blocks.append(f"===== {path}\n{error}")
    return clip("\n\n".join(blocks))


def tree_of(root: Path, path: str | None = None, depth: int = 3) -> str:
    """The shape of a directory, indented, without any of the contents.
    Cheap orientation before reading anything."""
    target = resolve(root, path)
    if not target.is_dir():
        raise ToolboxError(f"{path} is a file; read_file shows what is in it")
    rows: list[str] = []

    def walk(here: Path, level: int) -> None:
        if level > max(1, min(depth, 8)):
            return
        for child in sorted(here.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name.startswith("."):
                continue
            rows.append(f"{'  ' * (level - 1)}{child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir():
                walk(child, level + 1)

    walk(target, 1)
    return clip("\n".join(rows)) if rows else "the directory is empty"


def edit_text(root: Path, path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace exact text in a file. Refuses what it cannot do without
    guessing: text that is not there, or text that is there more than once
    when only one was meant."""
    target = resolve(root, path)
    if target.is_dir():
        raise ToolboxError(f"{path} is a directory")
    if not old:
        raise ToolboxError("the text to replace must not be empty; write_file replaces a file")
    text = target.read_text(encoding="utf-8", errors="replace")
    found = text.count(old)
    if found == 0:
        raise ToolboxError(f"that text does not appear in {path}; read it and match it exactly")
    if found > 1 and not replace_all:
        raise ToolboxError(
            f"that text appears {found} times in {path}; give more of its surroundings,"
            " or pass replace_all"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")
    return f"replaced {found} occurrence{'' if found == 1 else 's'} in {path}"


def move(root: Path, source: str, destination: str) -> str:
    """Move or rename, both ends inside the root."""
    origin = resolve(root, source)
    target = inside(root, destination)
    if target.exists():
        raise ToolboxError(f"{destination} already exists; it is not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    origin.rename(target)
    return f"moved {source} to {target.relative_to(root)}"


def write_text(root: Path, path: str, content: str) -> str:
    """Write a file, inside the root. Whether this call needed a signature
    was decided before it ran, by the application — never here."""
    target = inside(root, path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ToolboxError(f"cannot write {path}: {error}") from error
    return f"wrote {len(content.encode('utf-8'))} bytes to {target.relative_to(root)}"
