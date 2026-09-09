"""`/agent`: the scanned agents, the one in use marked."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets.option_list import Option

from cli.registry import AgentInfo
from cli.screens.chooser import Chooser


class AgentPicker(Chooser[str | None]):
    """The scanned agents, each with its source; the one in use marked,
    one that cannot load with the reason. Dismisses with the agent's
    name, or None."""

    TITLE_TEXT = "Select an agent"

    def __init__(
        self, entries: Sequence[AgentInfo], current: str, *, path: str = "~/.void/agents"
    ) -> None:
        super().__init__()
        self._entries = list(entries)
        self._current = current
        self.BLURB = (
            "The agent the next turns run — rebuilt every turn, so it takes effect at once."
            f" Your own is a build_agent(llm) in a .py under {path} or a --workspace folder:"
            " a new file shows on the next /agent, an edit needs a restart."
        )

    def rows(self) -> list[Option]:
        return [
            Option(
                Content.from_markup(
                    "$mark $n $name [$text-muted]$source[/] $blurb",
                    mark="●" if info.id == self._current else " ",
                    n=f"{index + 1}.".rjust(3),
                    name=info.id[:14].ljust(14),
                    source=info.source[:14].ljust(14),
                    blurb=(
                        Content.from_markup("[$error]cannot load: $why[/]", why=info.error)
                        if info.error is not None
                        else Content(info.blurb)
                    ),
                ),
                id=info.id,
            )
            for index, info in enumerate(self._entries)
        ]

    def initial(self) -> str | None:
        return self._current if any(info.id == self._current for info in self._entries) else None

    def chosen(self, option_id: str) -> str | None:
        return option_id
