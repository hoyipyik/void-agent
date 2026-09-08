"""The composer: a multi-line box under a prompt sign.

Enter sends. A new line is shift / alt / ctrl / cmd + Enter where the
terminal can tell them apart (the Kitty keyboard protocol: Ghostty,
Kitty, WezTerm, iTerm2), and a trailing backslash + Enter everywhere; a
pasted newline is kept. A paste that is a file's path — what a drag from
the file manager, or cmd+v with a copied file, arrives as — attaches the
file instead of inserting the path. ctrl+v or cmd+v asks the app for the
OS clipboard (an image, a copied file): a terminal paste never carries
one. Copy and cut are the TextArea's own (ctrl / cmd + c, x).

Some keys belong to the app: while the command menu is open, the arrows,
Tab and Escape; while an earlier message is being rewound to, the
arrows off the first and last line and Escape; and ↑ in an empty box,
which starts that rewind. They go up as `Navigate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.binding import BindingType
from textual.message import Message as UiMessage
from textual.widgets import TextArea

from cli.attachments import paths_in

MENU_KEYS = frozenset({"up", "down", "tab", "escape"})
REWIND_KEYS = frozenset({"up", "down", "escape"})
NEWLINE_KEYS = frozenset({"shift+enter", "alt+enter", "ctrl+enter", "super+enter", "ctrl+j"})


class Composer(TextArea):
    DEFAULT_CSS = """
    Composer {
        height: auto; min-height: 1; max-height: 10; width: 1fr;
        border: none; padding: 0; background: transparent;
    }
    Composer:focus { border: none; }
    Composer > .text-area--cursor-line { background: transparent; }
    Composer > .text-area--placeholder { color: $text-muted; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [("ctrl+v,super+v", "clipboard", "Paste file/image")]

    class Submitted(UiMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class PastedPaths(UiMessage):
        def __init__(self, paths: list[Path]) -> None:
            super().__init__()
            self.paths = paths

    class ClipboardRequested(UiMessage):
        pass

    class Navigate(UiMessage):
        """A key meant for the app: the command menu, or the rewind."""

        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    def __init__(self, placeholder: str) -> None:
        super().__init__(
            placeholder=placeholder,
            id="composer",
            soft_wrap=True,
            tab_behavior="focus",
            show_line_numbers=False,
        )
        # Both set by the app: the command menu is open above; an earlier
        # message is being rewound to.
        self.menu_open = False
        self.rewinding = False

    def _on_first_line(self) -> bool:
        return self.cursor_location[0] == 0

    def _on_last_line(self) -> bool:
        return self.cursor_location[0] == self.document.line_count - 1

    def _captured(self, key: str) -> bool:
        """Whether the app, not the box, takes this key now."""
        if self.menu_open:
            return key in MENU_KEYS
        if self.rewinding:
            if key == "up":
                return self._on_first_line()
            if key == "down":
                return self._on_last_line()
            return key in REWIND_KEYS
        return key == "up" and not self.text

    # Textual runs every `_on_<event>` down the MRO unless the event's
    # default is prevented, so these never call super(): what they do not
    # consume, TextArea's own handler takes next.

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if self._captured(key):
            event.stop()
            event.prevent_default()
            self.post_message(self.Navigate(key))
            return
        if key in NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key != "enter":
            return
        event.stop()
        event.prevent_default()
        if self.text.endswith("\\"):
            self.text = self.text[:-1] + "\n"
            self.move_cursor(self.document.end)
            return
        text = self.text.strip()
        if text:
            self.post_message(self.Submitted(text))
            self.clear()

    async def _on_paste(self, event: events.Paste) -> None:
        # Bubbling stops here either way: the app would hand a paste back
        # to the focused widget — this one — a second time.
        event.stop()
        paths = paths_in(event.text)
        if paths:
            event.prevent_default()
            self.post_message(self.PastedPaths(paths))

    def action_clipboard(self) -> None:
        self.post_message(self.ClipboardRequested())

    def load(self, text: str) -> None:
        """Put `text` in the box, the cursor at its end."""
        self.text = text
        self.move_cursor(self.document.end)
