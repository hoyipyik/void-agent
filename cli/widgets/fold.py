"""A line that folds open: the head, a one-line summary under `⎿`, and
on a click the whole payload. A tool call is one; a tool's own
`data-<kind>` progress is one."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Static

from cli.widgets.format import inline, pretty

STATE_COLOURS = {"output-available": "$success", "output-error": "$error"}


class Foldable(Vertical):
    """A line that folds open: the head, a one-line summary under `⎿`,
    and — on a click — the whole payload in the summary's place."""

    DEFAULT_CSS = """
    Foldable { height: auto; margin: 0 0 1 0; }
    Foldable .fold-head { height: auto; }
    Foldable .fold-summary { height: auto; color: $text; }
    Foldable .fold-body { height: auto; color: $text-muted; padding: 0 0 0 5; display: none; }
    Foldable.-open .fold-body { display: block; }
    Foldable.-open .fold-summary { display: none; }
    """

    collapsed: reactive[bool] = reactive(True)

    def __init__(self) -> None:
        super().__init__()
        self._head = Static("", classes="fold-head")
        self._summary = Static("", classes="fold-summary")
        self._body = Static("", classes="fold-body")

    def compose(self) -> ComposeResult:
        yield self._head
        yield self._summary
        yield self._body

    def on_mount(self) -> None:
        self.set_class(not self.collapsed, "-open")

    def watch_collapsed(self, collapsed: bool) -> None:
        self.set_class(not collapsed, "-open")

    def on_click(self) -> None:
        self.collapsed = not self.collapsed

    def show(self, head: Content, summary: Content, body: str) -> None:
        self._head.update(head)
        self._summary.update(summary)
        # Verbatim: a tool's data is not markup, whatever brackets it holds.
        self._body.update(Content(body))


class ToolChip(Foldable):
    """A tool call: `⏺ name(args)`, running, then its result — one line
    under `⎿`, the whole payload on a click. A failure opens itself, as
    the web's chip does."""

    def __init__(self, part: dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = ""
        self.state = ""
        self._read(part)
        if part.get("state") == "output-error":
            self.set_reactive(Foldable.collapsed, False)

    def _read(self, part: dict[str, Any]) -> None:
        self.tool_name = str(part.get("toolName", "?"))
        self.state = str(part.get("state", ""))
        colour = STATE_COLOURS.get(self.state, "$warning")
        head = Content.from_markup(
            f"[{colour}]⏺[/] [b]$name[/]([$text-muted]$args[/])",
            name=self.tool_name,
            args=inline(part.get("input", {})) if "input" in part else "",
        )
        match self.state:
            case "output-available":
                summary = Content.from_markup("  ⎿  $text", text=inline(part.get("output")))
            case "output-error":
                summary = Content.from_markup(
                    "  ⎿  [$error]$text[/]", text=inline(str(part.get("errorText", "")))
                )
            case _:
                summary = Content.from_markup("  ⎿  [$text-muted]running…[/]")
        self.show(head, summary, self._body_text(part))

    @staticmethod
    def _body_text(part: dict[str, Any]) -> str:
        sections: list[str] = []
        if "input" in part:
            sections.append(f"input\n{pretty(part['input'])}")
        match part.get("state"):
            case "output-available":
                sections.append(f"output\n{pretty(part.get('output'))}")
            case "output-error":
                sections.append(f"error\n{part.get('errorText', '')}")
            case _:
                pass
        return "\n\n".join(sections)

    def update(self, part: dict[str, Any]) -> None:
        self._read(part)
        if self.state == "output-error":
            self.collapsed = False


class DataCard(Foldable):
    """A tool's own progress, `data-<kind>`: shown by name, the payload
    folded."""

    def __init__(self, kind: str, data: Any) -> None:
        super().__init__()
        self.kind = kind
        self.show(
            Content.from_markup("[$secondary]◆[/] [b]$kind[/]", kind=kind),
            Content.from_markup("  ⎿  $text", text=inline(data)),
            pretty(data),
        )
