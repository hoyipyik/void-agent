"""`/session`: every saved session, newest first, and a way to start a
new one."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from textual.content import Content
from textual.widgets.option_list import Option

from cli.screens.chooser import Chooser
from cli.session import SessionSummary


def ago(when: datetime, now: datetime | None = None) -> str:
    """How long ago, in words a list can hold."""
    now = now or datetime.now(UTC)
    seconds = max(0, int((now - when).total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} h ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return when.astimezone().strftime("%Y-%m-%d")


class SessionPicker(Chooser[str | None]):
    """Every saved session, newest first, and a way to start a new one.
    Dismisses with the picked session's id, `NEW`, or None."""

    NEW = "new"
    TITLE_TEXT = "Resume a session"
    BLURB = "Saved under ~/.void/sessions; the one open now is marked."

    def __init__(self, sessions: Sequence[SessionSummary], current: str | None) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._current = current

    def rows(self) -> list[Option]:
        rows = [Option(Content.from_markup("[$primary]+[/]  New session"), id=self.NEW)]
        now = datetime.now(UTC)
        for session in self._sessions:
            rows.append(
                Option(
                    Content.from_markup(
                        "$mark [$text-muted]$when[/]  $title",
                        mark="●" if session.id == self._current else " ",
                        when=ago(session.updated_at, now).rjust(12),
                        title=session.title,
                    ),
                    id=session.id,
                )
            )
        return rows

    def chosen(self, option_id: str) -> str | None:
        return option_id
