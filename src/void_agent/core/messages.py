"""The conversation vocabulary.

Conversation history is NOT a runtime concept: it is a field of whichever
input type wants it (a chat server's `history: list[Message]`), rendered
into a run's transcript by that agent's `prompt` mapping like any other
typed input. The session that persists these messages is the durable state
of the system; the run itself keeps nothing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from void_agent.core.content import ContentPart, TextContent


class Role(enum.StrEnum):
    """A conversation side as the model sees it. The system prompt is not a
    role here — it travels separately, in the `Agent`."""

    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def parse_lossy(cls, raw: str) -> Role:
        """For roles read back from storage: anything that is not literally
        "user" collapses to assistant — the model only knows two sides."""
        return cls.USER if raw == "user" else cls.ASSISTANT


@dataclass(frozen=True, slots=True)
class Message:
    """One message: a role and one ordered sequence of content parts.

    A user message is text, images and PDFs in send order; assistant history
    is one text part. `Message.user` / `Message.assistant` build both, wrapping
    plain strings as `TextContent`.
    """

    role: Role
    content: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("message content must not be empty")
        if self.role is not Role.USER and not (
            len(self.content) == 1 and isinstance(self.content[0], TextContent)
        ):
            raise ValueError("assistant history is exactly one text part")

    @classmethod
    def user(cls, *parts: str | ContentPart) -> Message:
        """A user message; strings become `TextContent`, order is send order."""
        return cls(Role.USER, tuple(TextContent(p) if isinstance(p, str) else p for p in parts))

    @classmethod
    def assistant(cls, text: str) -> Message:
        """Replayed assistant history: one text."""
        return cls(Role.ASSISTANT, (TextContent(text),))
