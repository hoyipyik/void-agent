"""The two sides of the transcript: the assistant's text, a dot in the
gutter; the person's message after the prompt sign, a line per
attachment."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.widgets import Markdown, Static

from void_agent import parts_text


class Reply(Markdown):
    """The assistant's text — one per text part."""

    DEFAULT_CSS = """
    Reply { background: transparent; padding: 0; margin: 0; height: auto; width: 1fr; }
    Reply MarkdownH1, Reply MarkdownH2, Reply MarkdownH3, Reply MarkdownH4,
    Reply MarkdownH5, Reply MarkdownH6 {
        background: transparent; color: $primary; border: none;
        padding: 0; margin: 0; text-style: bold; text-align: left; content-align: left top;
    }
    Reply MarkdownFence {
        margin: 0 0 1 0; padding: 0 1; background: transparent;
        border: none; border-left: thick $border-blurred;
    }
    Reply MarkdownBlockQuote { border-left: outer $border-blurred; background: transparent; }
    Reply MarkdownBlock:dark > .code_inline, Reply MarkdownBlock:light > .code_inline {
        background: transparent; color: $secondary;
    }
    Reply > MarkdownParagraph { margin: 0 0 1 0; }
    Reply MarkdownListItem MarkdownParagraph { margin: 0; }
    Reply MarkdownBulletList, Reply MarkdownOrderedList { margin: 0 0 1 0; }
    """


class Said(Horizontal):
    """A reply with its dot in the gutter."""

    DEFAULT_CSS = """
    Said { height: auto; margin: 0 0 0 0; }
    Said .dot { width: 2; height: 1; color: $text-muted; }
    """

    def __init__(self, reply: Reply) -> None:
        super().__init__()
        self.reply = reply

    def compose(self) -> ComposeResult:
        yield Static("⏺", classes="dot")
        yield self.reply


class UserBubble(Static):
    """The person's message: its text after the prompt sign, and a line
    per attachment."""

    DEFAULT_CSS = """
    UserBubble { color: $text; padding: 0 1; margin: 1 0; height: auto; }
    UserBubble.-target { border-left: outer $primary; color: $text; }
    """

    def __init__(self, parts: list[dict[str, Any]]) -> None:
        text = parts_text(parts).strip()
        sign = "[$primary b]❯[/] $text"  # noqa: RUF001
        lines = [Content.from_markup(sign, text=text)] if text else []
        for part in parts:
            if part.get("type") == "file":
                name = str(part.get("filename") or "attachment")
                lines.append(
                    Content.from_markup(
                        "  ⎿  📎 $name [$text-muted]($kind)[/]",
                        name=name,
                        kind=str(part.get("mediaType", "?")),
                    )
                )
        super().__init__(Content("\n").join(lines), classes="user")
