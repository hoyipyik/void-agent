"""A list that stays open: its rows are switches, not a choice of one.
`/mcp` and `/skill` are the two boards built on it.

A mark is the person's word, made at the edge. No server and no model
ever sets one."""

from __future__ import annotations

from textual import events
from textual.content import Content

from cli.screens.chooser import Chooser

# What a switch shows, and what Enter cycles through. A tool has three
# marks; a server and a skill are simply on or off.
MARKS: dict[str, str] = {
    "on": "[✓] on  ",
    "signed": "[✎] sign",
    "off": "[ ] off ",
}
TOOL_CYCLE: dict[str, str] = {"on": "signed", "signed": "off", "off": "on"}
PLAIN_CYCLE: dict[str, str] = {"on": "off", "off": "on"}


class Switchboard(Chooser[dict[str, str] | None]):
    """A list that stays open: Enter or Space cycles the highlighted row's
    mark, Escape closes with every row's state. Unlike the other choosers
    nothing here is a choice of one — the rows are switches.

    A mark is the person's word, made at the edge. No server and no model
    ever sets one."""

    HINT = "↑↓ move · Enter/Space cycle · Esc done"

    def __init__(self) -> None:
        super().__init__()
        self._states: dict[str, str] = {}
        self._initial: dict[str, str] = {}

    def start(self, states: dict[str, str]) -> None:
        """The states the board opens on. What it dismisses with is only
        what moved: a row the person did not touch must stay unwritten, or
        closing the board would freeze today's defaults into the config and
        `mcp.json` would stop meaning anything."""
        self._states = states
        self._initial = dict(states)

    @property
    def changed(self) -> dict[str, str]:
        return {key: state for key, state in self._states.items() if state != self._initial[key]}

    def cycle_for(self, option_id: str) -> dict[str, str]:
        """The marks this row moves through."""
        return PLAIN_CYCLE

    def render_row(self, option_id: str) -> Content:
        raise NotImplementedError

    def chosen(self, option_id: str) -> dict[str, str] | None:
        return self.changed

    def regroups(self, option_id: str) -> bool:
        """Whether this row's mark changes which rows exist at all — a
        switch that turns a group on or off."""
        return False

    def selected(self, option_id: str) -> None:
        state = self._states.get(option_id)
        if state is None:
            return
        self._states[option_id] = self.cycle_for(option_id)[state]
        if not self.regroups(option_id):
            self.choices.replace_option_prompt(option_id, self.render_row(option_id))
            return
        # The rows under it are gone or back; the switch itself keeps the
        # highlight, so the next press lands where the eye is.
        choices = self.choices
        choices.clear_options()
        choices.add_options(self.rows())
        choices.highlighted = choices.get_option_index(option_id)

    def on_key(self, event: events.Key) -> None:
        if event.key == "space":
            event.stop()
            self.choices.action_select()
            return
        super().on_key(event)

    def action_cancel(self) -> None:
        self.dismiss(self.changed)
