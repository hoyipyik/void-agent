"""The shelf: what a folder of skills holds, and the tools that read it.

Frontmatter is read at mount — it becomes the model's routing table, so a
skill that says nothing about itself is refused there and then. The
instructions themselves are read at the call, so editing a SKILL.md lands
on the next turn rather than the next restart.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from void_agent.core.errors import Rejected
from void_agent.core.tool import Tool

SKILL_FILE = "SKILL.md"
PREFIX = "skill__"
READER = f"{PREFIX}file"
FRONTMATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)


class SkillFolderError(Exception):
    """A folder that means to be a skill but cannot be read as one."""


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """One skill, as the person sees it and the model is told of it."""

    id: str
    name: str
    description: str
    path: Path
    files: tuple[str, ...]

    @property
    def tool_name(self) -> str:
        return f"{PREFIX}{_identifier(self.id)}"


class NoInput(BaseModel):
    """A skill takes no arguments: reading it is the whole of it."""

    model_config = ConfigDict(extra="forbid")


class FileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: str
    file: str


def read_skills(folder: Path | str) -> tuple[SkillInfo, ...]:
    """Every skill on the shelf, in name order. A folder that is absent is
    simply no skills; one that holds a SKILL.md saying nothing about
    itself is an error, because that line is the model's only way to know
    when to reach for it."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return ()
    found: list[SkillInfo] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_file = directory / SKILL_FILE
        if not skill_file.is_file():
            continue
        found.append(_info(directory, skill_file))
    return tuple(found)


def skills_in(folder: Path | str, *, only: frozenset[str] | None = None) -> tuple[Tool, ...]:
    """The shelf as tools. `only` keeps the named skills and drops the
    rest — a caller holding the infos already should use `tools_for`."""
    kept = [info for info in read_skills(folder) if only is None or info.id in only]
    return tools_for(kept)


def tools_for(skills: Sequence[SkillInfo]) -> tuple[Tool, ...]:
    """One tool per skill, plus the single reader for the files they carry
    — and no reader at all when none of them carries one."""
    tools = [_skill_tool(info) for info in skills]
    if any(info.files for info in skills):
        tools.append(_reader(skills))
    return tuple(tools)


def _info(directory: Path, skill_file: Path) -> SkillInfo:
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError as error:
        raise SkillFolderError(f"cannot read {directory.name}/{SKILL_FILE}: {error}") from error
    front = _frontmatter(text)
    description = front.get("description", "").strip()
    if not description:
        raise SkillFolderError(
            f"the skill `{directory.name}` has no description in its {SKILL_FILE}"
            " frontmatter — that line is how the model knows when to read it"
        )
    return SkillInfo(
        id=directory.name,
        name=front.get("name", "").strip() or directory.name,
        description=description,
        path=skill_file,
        files=_bundled(directory),
    )


def _frontmatter(text: str) -> dict[str, str]:
    """The `key: value` lines at the top. Deliberately not YAML: a skill's
    front matter is a name and a sentence, and a parser is a dependency."""
    match = FRONTMATTER.match(text)
    if match is None:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and not key.startswith(("#", " ")):
            fields[key.strip().lower()] = value.strip().strip("\"'")
    return fields


def _body(skill_file: Path) -> str:
    text = skill_file.read_text(encoding="utf-8")
    return FRONTMATTER.sub("", text, count=1).strip()


def _bundled(directory: Path) -> tuple[str, ...]:
    return tuple(
        sorted(p.name for p in directory.iterdir() if p.is_file() and p.name != SKILL_FILE)
    )


def _identifier(name: str) -> str:
    """A folder name as a tool name: what a provider's schema accepts."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower() or "skill"


def _skill_tool(info: SkillInfo) -> Tool:
    """One skill's instructions, read out when the model asks for them."""

    async def handler(input: NoInput) -> str:
        body = _body(info.path)
        if not info.files:
            return body
        listed = ", ".join(info.files)
        return f"{body}\n\nThis skill also carries: {listed} — read one with `{READER}`."

    return Tool(
        name=info.tool_name,
        description=f"Read the instructions for: {info.description}",
        handler=handler,
        input_schema=NoInput.model_json_schema(),
    )


def _reader(kept: Iterable[SkillInfo]) -> Tool:
    """The one way into a skill's own files, and no further. A path that
    resolves outside the skill it names is refused — the model can ask for
    anything, so the folder, not the argument, is the boundary."""
    directories = {info.id: info.path.parent.resolve() for info in kept}
    named = ", ".join(f"{info.id} ({', '.join(info.files)})" for info in kept if info.files)

    async def handler(input: FileRequest) -> str:
        directory = directories.get(input.skill)
        if directory is None:
            raise Rejected(f"no skill named {input.skill}; there are: {', '.join(directories)}")
        target = (directory / input.file).resolve()
        if not target.is_relative_to(directory) or not target.is_file():
            raise Rejected(f"{input.skill} carries no file named {input.file}")
        return target.read_text(encoding="utf-8")

    return Tool(
        name=READER,
        description=f"Read a file a skill carries. Available: {named}.",
        handler=handler,
        input_schema=FileRequest.model_json_schema(),
    )
