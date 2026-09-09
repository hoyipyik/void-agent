"""What an empty session shows: the logo (VoidAgent, figlet's slant
face), the ways in, the agent and the model. The transcript's header,
drawn once at the top of the log and left there — replayed above a
resumed session too — its lines kept current: `/model` and `/agent`
change what the next turn runs on, and the box says so."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from cli.config import Config
from cli.labels import model_label, tilde

# The logo, two words in two colours, row by row (figlet's slant face).
LOGO_VOID = (
    " _    __      _     __",
    "| |  / /___  (_)___/ /",
    "| | / / __ \\/ / __  /",
    "| |/ / /_/ / / /_/ /",
    "|___/\\____/_/\\__,_/",
    "",
)
LOGO_AGENT = (
    "    ___                    __",
    "   /   | ____ ____  ____  / /_",
    "  / /| |/ __ `/ _ \\/ __ \\/ __/",
    " / ___ / /_/ /  __/ / / / /_",
    "/_/  |_\\__, /\\___/_/ /_/\\__/",
    "      /____/",
)
LOGO_WIDTH = max(len(row) for row in LOGO_VOID) + 1 + max(len(row) for row in LOGO_AGENT)


def logo() -> Content:
    """VoidAgent, the first word in the primary colour, the second in
    the secondary — one Content, the rows joined."""
    width = max(len(row) for row in LOGO_VOID)
    rows = [
        Content.from_markup(
            "[$primary b]$void[/] [$secondary b]$agent[/]", void=void.ljust(width), agent=agent
        )
        for void, agent in zip(LOGO_VOID, LOGO_AGENT, strict=True)
    ]
    return Content("\n").join(rows)


class Welcome(Vertical):
    """What an empty session shows: the logo, the ways in, the model."""

    DEFAULT_CSS = """
    Welcome {
        border: round $primary; padding: 0 2; margin: 1 0;
        width: auto; max-width: 76; height: auto;
    }
    Welcome .logo { width: auto; height: auto; text-wrap: nowrap; text-overflow: clip; }
    Welcome .lines { width: 1fr; height: auto; margin-top: 1; }
    """

    def __init__(self, config: Config, home: Path, agent: str) -> None:
        super().__init__()
        self._home = home
        lines = self._lines(config, agent)
        # What the box says under the logo, in plain words.
        self.details = lines.plain
        self._detail = Static(lines, classes="lines")

    def _lines(self, config: Config, agent: str) -> Content:
        lines = [
            "  [$primary]/help[/] commands   [$primary]/model[/] model   [$primary]/agent[/] agent"
            "   [$primary]/session[/] resume",
            "",
            "  [$text-muted]agent[/]  $agent",
            "  [$text-muted]model[/]  $model",
            "  [$text-muted]home[/]   $home",
        ]
        if not config.configured():
            lines.append("")
            lines.append("  [$warning]no API key yet[/] — [$primary]/key[/] adds one")
        return Content.from_markup(
            "\n".join(lines), agent=agent, model=model_label(config), home=tilde(self._home)
        )

    def show(self, config: Config, agent: str) -> None:
        """The agent and the model as they stand now; the key warning
        goes once a key is saved."""
        lines = self._lines(config, agent)
        self.details = lines.plain
        self._detail.update(lines)

    def compose(self) -> ComposeResult:
        yield Static(logo(), classes="logo")
        yield self._detail
