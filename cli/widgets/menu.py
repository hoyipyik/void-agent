"""The slash-command menu that unfolds above the prompt as "/" is
typed."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from cli.commands import Spec


class CommandMenu(OptionList):
    """The slash commands, listed as the name is typed: the arrow keys
    move, Tab completes, Enter runs, a click runs. The composer keeps
    the focus; the app relays the keys."""

    DEFAULT_CSS = """
    CommandMenu {
        display: none; height: auto; max-height: 13;
        margin: 0 1; padding: 0 1; border: round $border-blurred;
    }
    CommandMenu.-open { display: block; }
    CommandMenu > .option-list--option { padding: 0 1; }
    CommandMenu > .option-list--option-highlighted {
        background: $block-cursor-background; color: $block-cursor-foreground; text-style: none;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="menu")
        self._specs: dict[str, Spec] = {}
        self.open = False

    @staticmethod
    def _row(spec: Spec) -> Content:
        return Content.from_markup(
            "[$primary]$name[/]  [$text-muted]$summary[/]",
            name=spec.usage.ljust(18),
            summary=spec.summary,
        )

    def show(self, specs: Sequence[Spec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self.set_options([Option(self._row(spec), id=spec.name) for spec in specs])
        self.highlighted = 0 if specs else None
        self.open = bool(specs)
        self.set_class(self.open, "-open")

    def hide(self) -> None:
        if not self.open:
            return
        self.open = False
        self.remove_class("-open")
        self.clear_options()
        self._specs = {}

    def spec_named(self, name: str | None) -> Spec | None:
        return self._specs.get(name or "")

    @property
    def chosen(self) -> Spec | None:
        if not self.open or self.highlighted is None:
            return None
        return self.spec_named(self.get_option_at_index(self.highlighted).id)

    def move(self, key: str) -> None:
        if key == "up":
            self.action_cursor_up()
        elif key == "down":
            self.action_cursor_down()
