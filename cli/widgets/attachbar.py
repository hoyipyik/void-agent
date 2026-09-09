"""The attachments waiting for the next message, in a line above the
prompt — hidden while there are none. The list lives here: the shell
adds to it as a path is dragged in, an `@path` is written or the
clipboard is read, and takes it when the message goes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from textual.content import Content
from textual.widgets import Static

from cli.session.attachments import Attachment


class AttachmentBar(Static):
    DEFAULT_CSS = """
    AttachmentBar { display: none; color: $text-muted; margin: 0 2; height: auto; }
    AttachmentBar.-shown { display: block; }
    """

    def __init__(self) -> None:
        super().__init__("", id="attachments")
        self.pending: list[Attachment] = []

    def add(self, attachments: Iterable[Attachment]) -> None:
        self.pending.extend(attachments)
        self._show()

    def drop(self) -> None:
        self.pending = []
        self._show()

    def parts(self) -> list[dict[str, Any]]:
        """The `file` parts the next message carries, in order."""
        return [attachment.part() for attachment in self.pending]

    def _show(self) -> None:
        if not self.pending:
            self.update("")
            self.remove_class("-shown")
            return
        chips = Content("  ").join(
            Content.from_markup("📎 $name [$text-muted]($size)[/]", name=a.name, size=a.size)
            for a in self.pending
        )
        self.update(
            chips
            + Content.from_markup(
                "   [$text-muted]sent with the next message · /detach drops them[/]"
            )
        )
        self.add_class("-shown")
