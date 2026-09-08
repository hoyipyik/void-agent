"""Skills: a folder of instructions becomes tools that read them out.

A skill carries knowledge, never a capability — calling one puts text in
the transcript and nothing else happens."""

from __future__ import annotations

from pathlib import Path

import pytest

from void_agent import EventSender, Rejected
from void_agent.skills import SkillFolderError, read_skills, skills_in

PI = """---
name: PI drafting
description: how this company writes a proforma invoice
---

Always quote in USD. Put the incoterm on line 1.
"""

REFUNDS = """---
description: when a refund may be issued without a manager
---

Under 500 USD and within 30 days: issue it.
"""


def shelf(root: Path) -> Path:
    (root / "pi-drafting").mkdir(parents=True)
    (root / "pi-drafting" / "SKILL.md").write_text(PI, encoding="utf-8")
    (root / "pi-drafting" / "templates.md").write_text("# PI template\n", encoding="utf-8")
    (root / "refunds").mkdir()
    (root / "refunds" / "SKILL.md").write_text(REFUNDS, encoding="utf-8")
    return root


def test_a_skill_is_read_from_its_frontmatter(tmp_path: Path) -> None:
    pi, refunds = read_skills(shelf(tmp_path))
    assert (pi.id, pi.name) == ("pi-drafting", "PI drafting")
    assert pi.description == "how this company writes a proforma invoice"
    assert pi.files == ("templates.md",)
    # A skill that names no name is known by its folder.
    assert (refunds.id, refunds.name) == ("refunds", "refunds")
    assert refunds.files == ()


def test_a_folder_without_a_skill_file_is_not_a_skill(tmp_path: Path) -> None:
    shelf(tmp_path)
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "README.md").write_text("nothing here", encoding="utf-8")
    assert [info.id for info in read_skills(tmp_path)] == ["pi-drafting", "refunds"]


def test_a_missing_folder_is_simply_no_skills(tmp_path: Path) -> None:
    assert read_skills(tmp_path / "absent") == ()


def test_a_skill_without_a_description_is_refused_at_the_door(tmp_path: Path) -> None:
    (tmp_path / "vague").mkdir(parents=True)
    (tmp_path / "vague" / "SKILL.md").write_text(
        "---\nname: Vague\n---\n\nstuff", encoding="utf-8"
    )
    with pytest.raises(SkillFolderError, match="vague"):
        read_skills(tmp_path)


async def test_calling_a_skill_reads_its_instructions_out(tmp_path: Path) -> None:
    pi, _refunds, _reader = skills_in(shelf(tmp_path))
    assert pi.name == "skill__pi_drafting"
    assert pi.description.startswith("Read the instructions for")
    assert "how this company writes" in pi.description
    body = await pi.invoke({}, EventSender())
    assert "Always quote in USD" in body
    assert "---" not in body  # the frontmatter is not the instructions
    assert "templates.md" in body  # what else the skill carries


async def test_the_instructions_are_read_when_they_are_asked_for(tmp_path: Path) -> None:
    """Editing a SKILL.md takes effect on the next call, not the next
    restart."""
    pi = skills_in(shelf(tmp_path))[0]
    (tmp_path / "pi-drafting" / "SKILL.md").write_text(PI.replace("USD", "EUR"), encoding="utf-8")
    assert "EUR" in await pi.invoke({}, EventSender())


async def test_a_skill_carries_no_capability_only_words(tmp_path: Path) -> None:
    pi = skills_in(shelf(tmp_path))[0]
    assert pi.input_schema.get("properties", {}) == {}


async def test_a_bundled_file_is_read_through_the_one_reader(tmp_path: Path) -> None:
    tools = skills_in(shelf(tmp_path))
    reader = tools[-1]
    assert reader.name == "skill__file"
    assert "# PI template" in await reader.invoke(
        {"skill": "pi-drafting", "file": "templates.md"}, EventSender()
    )


async def test_the_reader_never_leaves_the_skills_folder(tmp_path: Path) -> None:
    reader = skills_in(shelf(tmp_path))[-1]
    for attempt in ("../../etc/passwd", "/etc/passwd", "../refunds/SKILL.md"):
        with pytest.raises(Rejected):
            await reader.invoke({"skill": "pi-drafting", "file": attempt}, EventSender())


async def test_a_shelf_with_no_bundled_files_needs_no_reader(tmp_path: Path) -> None:
    (tmp_path / "refunds").mkdir(parents=True)
    (tmp_path / "refunds" / "SKILL.md").write_text(REFUNDS, encoding="utf-8")
    (only,) = skills_in(tmp_path)
    assert only.name == "skill__refunds"


def test_only_the_named_skills_are_built(tmp_path: Path) -> None:
    tools = skills_in(shelf(tmp_path), only=frozenset({"refunds"}))
    assert [capability.name for capability in tools] == ["skill__refunds"]
