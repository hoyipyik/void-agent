"""`/agent`: the mounted agents, the one in use marked."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets.option_list import Option

from cli.agents import AgentInfo
from cli.screens.chooser import Chooser


class AgentPicker(Chooser[str | None]):
    """The mounted agents, the one in use marked. Dismisses with the
    agent's name, or None."""

    TITLE_TEXT = "Select an agent"
    BLURB = (
        "The agent the next turns run — rebuilt every turn, so it takes effect at once."
        " Mount your own at start: --agent module:function."
    )

    def __init__(self, entries: Sequence[AgentInfo], current: str) -> None:
        super().__init__()
        self._entries = list(entries)
        self._current = current

    def rows(self) -> list[Option]:
        return [
            Option(
                Content.from_markup(
                    "$mark $n $name [$text-muted]$blurb[/]",
                    mark="●" if info.id == self._current else " ",
                    n=f"{index + 1}.".rjust(3),
                    name=info.name.ljust(11),
                    blurb=info.blurb,
                ),
                id=info.id,
            )
            for index, info in enumerate(self._entries)
        ]

    def initial(self) -> str | None:
        return self._current if any(info.id == self._current for info in self._entries) else None

    def chosen(self, option_id: str) -> str | None:
        return option_id
