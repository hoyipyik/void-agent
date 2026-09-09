"""The prompt frame — a sign and the composer in one rounded box — and
the status line under it, which spins while a turn runs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

from cli.widgets.composer import Composer
from void_agent import (
    AgentEvent,
    AskIssued,
    TextDelta,
    TextStart,
    ToolInputAvailable,
    ToolInputStart,
)

SPINNER = "·✢✳✶✻✽✻✶✳✢"
IDLE_HINT = "/ commands · @path attaches · ⇧⏎ new line · ↑ edit earlier · esc stops"
REWIND_HINT = (
    "editing an earlier message · Enter replaces it and all after · ↑↓ move · Esc cancels"
)


class PromptFrame(Horizontal):
    """The prompt: a sign and the composer in one rounded box, brighter
    while it has the focus, dimmer while a turn runs."""

    DEFAULT_CSS = """
    PromptFrame {
        height: auto; margin: 0 1; padding: 0 1;
        border: round $border-blurred;
    }
    PromptFrame:focus-within { border: round $primary; }
    PromptFrame.-busy { border: round $border-blurred; }
    PromptFrame .sign { width: 2; height: 1; color: $primary; text-style: bold; }
    PromptFrame.-busy .sign { color: $text-muted; }
    """

    def __init__(self, composer: Composer) -> None:
        super().__init__(id="prompt")
        self._sign = Static("❯", classes="sign")  # noqa: RUF001
        self.composer = composer

    def compose(self) -> ComposeResult:
        yield self._sign
        yield self.composer

    def set_sign(self, sign: str) -> None:
        self._sign.update(Content.from_markup(sign))


class StatusBar(Horizontal):
    """One line under the prompt: a hint — or, while a turn runs, a
    spinner and what the turn is doing — and the model on the right."""

    DEFAULT_CSS = """
    StatusBar { height: 1; margin: 0 2 1 2; color: $text-muted; }
    StatusBar .left { width: 1fr; height: 1; }
    StatusBar .right { width: auto; height: 1; color: $text-muted; }
    """

    def __init__(self) -> None:
        super().__init__(id="status")
        self._left = Static(
            Content.from_markup("[$text-muted]$hint[/]", hint=IDLE_HINT), classes="left"
        )
        self._right = Static("", classes="right")
        # What the left side says now, in plain words.
        self.line = IDLE_HINT
        self._activity: str | None = None
        self._frame = 0
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield self._left
        yield self._right

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.12, self._tick, pause=True)

    def show_model(self, text: str) -> None:
        self._right.update(Content.from_markup("[$text-muted]$text[/]", text=text))

    def busy(self, activity: str) -> None:
        self._activity = activity
        if self._timer is not None:
            self._timer.resume()
        self._tick()

    def idle(self) -> None:
        self._activity = None
        if self._timer is not None:
            self._timer.pause()
        self.line = IDLE_HINT
        self._left.update(Content.from_markup("[$text-muted]$hint[/]", hint=IDLE_HINT))

    def show_hint(self, text: str) -> None:
        """A hint in the idle line's place — while a mode lasts."""
        self.line = text
        self._left.update(Content.from_markup("[$primary]$hint[/]", hint=text))

    def _tick(self) -> None:
        if self._activity is None:
            return
        self._frame = (self._frame + 1) % len(SPINNER)
        self.line = self._activity
        self._left.update(
            Content.from_markup(
                "[$primary]$frame[/] $activity  [$text-muted](esc to interrupt)[/]",
                frame=SPINNER[self._frame],
                activity=self._activity,
            )
        )


def activity(event: AgentEvent) -> str | None:
    """What the status line says the turn is doing, from the event."""
    match event:
        case ToolInputStart(tool_name=name) | ToolInputAvailable(tool_name=name):
            return f"Running {name}…"
        case TextStart() | TextDelta():
            return "Writing…"
        case AskIssued():
            return "Waiting for you…"
        case _:
            return None
