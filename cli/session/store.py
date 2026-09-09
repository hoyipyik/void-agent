"""Sessions on disk: one JSON file per session, holding UIMessage `parts`
verbatim — the shape the reference server's store keeps, so a session
written here is a conversation there. `Session.history()` is the
model-facing projection (`context_content`, attachments intact) of the
tail of the context window, exactly what the server's `context_messages`
hands the agent. The session is the durable state; the app keeps nothing
else between turns."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from void_agent import NO_USAGE, Message, Role, Usage, context_content, parts_text, usages

CONTEXT_MESSAGES = 20
TITLE_MAX = 120
UNTITLED = "New conversation"


def derive_title(text: str) -> str:
    trimmed = text.strip()
    return trimmed[:TITLE_MAX] if trimmed else UNTITLED


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    parts: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    id: str
    title: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Tally:
    """What the session has cost so far, read off its `data-usage` parts:
    the sum over every round-trip (`total`, the bill), how many there were
    (`steps`), and the latest one's prompt (`context`) — the size of what
    the model read last, which is what the next turn grows from."""

    total: Usage = NO_USAGE
    steps: int = 0
    context: int = 0

    @property
    def consumed(self) -> int:
        """Everything the session has spent, in and out together — the
        one number the status line carries."""
        return self.total.input + self.total.output


@dataclass(slots=True)
class Session:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[StoredMessage] = field(default_factory=list[StoredMessage])

    @classmethod
    def fresh(cls) -> Session:
        now = _now()
        return cls(id=uuid.uuid4().hex, title=UNTITLED, created_at=now, updated_at=now)

    def append(self, role: str, parts: list[dict[str, Any]]) -> StoredMessage:
        """One side of a turn. The first user message names the session."""
        message = StoredMessage(id=uuid.uuid4().hex, role=role, parts=parts)
        if not self.messages and role == "user":
            self.title = derive_title(parts_text(parts))
        self.messages.append(message)
        self.updated_at = _now()
        return message

    def clear(self) -> None:
        self.messages.clear()
        self.title = UNTITLED
        self.updated_at = _now()

    def truncate(self, index: int) -> None:
        """Drop the message at `index` and everything after it — the
        rewind before an earlier message is replaced."""
        del self.messages[index:]
        if not self.messages:
            self.title = UNTITLED
        self.updated_at = _now()

    def user_indexes(self) -> list[int]:
        return [i for i, message in enumerate(self.messages) if message.role == "user"]

    def history(
        self, limit: int = CONTEXT_MESSAGES, *, tool_output_limit: int | None = None
    ) -> list[Message]:
        """What the next turn's model reads: the context tail, each message
        projected through `context_content` — tool results, asks, answers,
        and the user's attachments all speak. A message with nothing to say
        is left out. `tool_output_limit` caps one tool result, in
        characters (`Config.tool_output_limit`); None keeps it whole."""
        return [
            Message(Role.parse_lossy(message.role), content)
            for message in self.messages[-limit:]
            if (content := context_content(message.parts, tool_output_limit=tool_output_limit))
        ]

    def summary(self) -> SessionSummary:
        return SessionSummary(self.id, self.title, self.updated_at)

    def tally(self) -> Tally:
        """The session's account, over every message it kept."""
        reported = [usage for message in self.messages for usage in usages(message.parts)]
        total = NO_USAGE
        for usage in reported:
            total = total + usage
        return Tally(
            total=total, steps=len(reported), context=reported[-1].input if reported else 0
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "messages": [{"id": m.id, "role": m.role, "parts": m.parts} for m in self.messages],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Session:
        messages = [
            StoredMessage(
                id=str(raw.get("id", "")),
                role=str(raw.get("role", "assistant")),
                parts=cast("list[dict[str, Any]]", raw.get("parts") or []),
            )
            for raw in cast("list[dict[str, Any]]", data.get("messages") or [])
        ]
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or UNTITLED),
            created_at=datetime.fromisoformat(str(data["createdAt"])),
            updated_at=datetime.fromisoformat(str(data["updatedAt"])),
            messages=messages,
        )


class SessionStore:
    """A directory of session files. A session with no messages owns no
    file: saving one removes what was there, so `/clear` leaves nothing
    behind and a fresh session is not listed until it says something."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def new(self) -> Session:
        return Session.fresh()

    def _path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.json"

    def save(self, session: Session) -> None:
        path = self._path(session.id)
        if not session.messages:
            path.unlink(missing_ok=True)
            return
        self._root.mkdir(parents=True, exist_ok=True)
        staged = path.with_suffix(".json.tmp")
        staged.write_text(json.dumps(session.to_json(), ensure_ascii=False), encoding="utf-8")
        staged.replace(path)

    def load(self, session_id: str) -> Session:
        data = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        return Session.from_json(cast("dict[str, Any]", data))

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list(self) -> list[SessionSummary]:
        """Every saved session, newest first."""
        if not self._root.exists():
            return []
        summaries: list[SessionSummary] = []
        for path in self._root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.from_json(cast("dict[str, Any]", data))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # a file that is not a session is not the app's to explain
            summaries.append(session.summary())
        summaries.sort(key=lambda s: (s.updated_at, s.id), reverse=True)
        return summaries
