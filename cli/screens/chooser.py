"""The shape every modal here takes: a titled list read with the keys —
the arrows move, Enter chooses, a digit jumps to the nth choosable row,
Escape cancels — and the dialog CSS they share."""

from __future__ import annotations

from typing import ClassVar, TypeVar

from textual import events
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

DIALOG_CSS = """
ModalScreen { align: center middle; background: transparent; }
#dialog {
    width: 78; height: auto; max-height: 90%;
    border: round $primary; background: $background; padding: 1 2;
}
#dialog .title { text-style: bold; color: $primary; }
#dialog .blurb { color: $text-muted; margin-bottom: 1; height: auto; }
#dialog OptionList {
    height: auto; max-height: 24; background: transparent; border: none; padding: 0;
}
#dialog OptionList:focus { border: none; }
#dialog OptionList > .option-list--option { padding: 0 1; }
#dialog OptionList > .option-list--option-highlighted {
    background: $block-cursor-background; color: $block-cursor-foreground; text-style: none;
}
#dialog OptionList > .option-list--option-disabled { color: $primary; text-style: bold; }
#dialog Input { margin-top: 1; border: round $border-blurred; background: $background; }
#dialog Input:focus { border: round $primary; }
#dialog .hint { color: $text-muted; margin-top: 1; }
"""
DIGITS = frozenset("123456789")

T = TypeVar("T")


class Chooser(ModalScreen[T]):
    """A titled list: the arrows move, Enter chooses, a digit jumps to
    the nth choosable row, Escape cancels."""

    CSS = DIALOG_CSS
    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]
    TITLE_TEXT = ""
    BLURB = ""
    HINT = "↑↓ move · Enter choose · 1-9 jump · Esc cancel"

    def rows(self) -> list[Option]:
        raise NotImplementedError

    def chosen(self, option_id: str) -> T:
        raise NotImplementedError

    def initial(self) -> str | None:
        """The id of the row highlighted first; the first choosable one
        when None."""
        return None

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.TITLE_TEXT, classes="title")
            if self.BLURB:
                yield Static(self.BLURB, classes="blurb")
            yield OptionList(*self.rows(), id="choices")
            yield Static(
                Content.from_markup("[$text-muted]$hint[/]", hint=self.HINT), classes="hint"
            )

    @property
    def choices(self) -> OptionList:
        return self.query_one("#choices", OptionList)

    def _choosable(self) -> list[int]:
        choices = self.choices
        return [
            index
            for index in range(choices.option_count)
            if not choices.get_option_at_index(index).disabled
        ]

    def on_mount(self) -> None:
        choices = self.choices
        wanted = self.initial()
        index = None
        if wanted is not None:
            try:
                index = choices.get_option_index(wanted)
            except OptionDoesNotExist:
                index = None
        if index is None:
            choosable = self._choosable()
            index = choosable[0] if choosable else None
        choices.highlighted = index
        choices.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # Textual runs the handler of every class in the MRO, so the row's
        # meaning lives in `selected` — a plain method a subclass overrides.
        event.stop()
        if event.option.id is not None:
            self.selected(event.option.id)

    def selected(self, option_id: str) -> None:
        """What a chosen row does: dismiss with it. A list of switches
        overrides this and stays open."""
        self.dismiss(self.chosen(option_id))

    def on_key(self, event: events.Key) -> None:
        if event.key not in DIGITS:
            return
        choosable = self._choosable()
        index = int(event.key) - 1
        if index >= len(choosable):
            return
        event.stop()
        self.choices.highlighted = choosable[index]
        self.choices.action_select()

    def action_cancel(self) -> None:
        self.dismiss(None)  # pyright: ignore[reportArgumentType]
