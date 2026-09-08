"""`/skill`: the skills on the shelf, each on or off."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets.option_list import Option

from cli.config import Config
from cli.screens.switchboard import MARKS, Switchboard
from void_agent.skills import SkillInfo

NO_SKILLS = (
    "No skills yet. A skill is a folder under {path} with a SKILL.md: `name` and"
    " `description` in its frontmatter, the instructions below it. Only the"
    " description stays in context — the rest is read when the model asks for it."
)


class SkillPicker(Switchboard):
    """The skills on the shelf, each on or off. A skill carries knowledge,
    never a capability, so there is nothing here to sign — turning one on
    puts its one-line description in the model's reach, and the
    instructions arrive only when it asks."""

    TITLE_TEXT = "Skills"

    def __init__(
        self,
        skills: Sequence[SkillInfo],
        config: Config,
        *,
        path: str = "~/.void/skills",
    ) -> None:
        super().__init__()
        self._skills = list(skills)
        self.start({info.id: config.skill_state(info.id) for info in self._skills})
        self.BLURB = (
            NO_SKILLS.format(path=path)
            if not self._skills
            else "Only the description stays in context; the instructions are read"
            " when the model asks for them."
        )

    def rows(self) -> list[Option]:
        return [Option(self.render_row(info.id), id=info.id) for info in self._skills]

    def render_row(self, option_id: str) -> Content:
        info = next(info for info in self._skills if info.id == option_id)
        return Content.from_markup(
            "  $mark  $name [$text-muted]$blurb[/]",
            mark=MARKS[self._states[option_id]],
            name=info.name.ljust(20),
            blurb=info.description.splitlines()[0][:46],
        )
