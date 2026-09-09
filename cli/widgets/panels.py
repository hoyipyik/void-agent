"""A titled box in the log — what `/help` and `/status` leave."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets import Static

from cli.commands import COMMANDS
from cli.config import Config
from cli.labels import model_label
from cli.providers.catalog import KEYED, PROVIDER_LABELS, ModelInfo
from cli.providers.ollama import alias


class Panel(Static):
    """A titled box in the log — what `/help` and `/status` leave."""

    DEFAULT_CSS = """
    Panel {
        border: round $border-blurred; padding: 0 2; margin: 1 0;
        width: auto; max-width: 80; height: auto;
    }
    """

    def __init__(self, title: str, body: Content) -> None:
        heading = Content.from_markup("[$primary b]$title[/]\n\n", title=title)
        super().__init__(heading + body)


def help_panel() -> Panel:
    rows = [
        Content.from_markup(
            "  [$primary]$usage[/]  [$text-muted]$summary[/]$aliases",
            usage=spec.usage.ljust(18),
            summary=spec.summary,
            aliases=f"  (also /{', /'.join(spec.aliases)})" if spec.aliases else "",
        )
        for spec in COMMANDS
    ]
    keys = Content.from_markup(
        "\n[b]keys[/]\n"
        "  [$primary]Enter[/]               send · with the menu open, run the command\n"
        "  [$primary]⇧ ⌥ ⌘ + Enter[/]       a new line (where the terminal can tell; "
        "[$primary]\\ + Enter[/] everywhere)\n"
        "  [$primary]Tab[/]                 complete the command under the cursor\n"
        "  [$primary]↑ ↓[/]                 move in a menu or on a card · [$primary]1-9[/] jump\n"
        "  [$primary]↑[/] in an empty box    edit an earlier message: Enter replaces it and"
        " all after, Esc cancels\n"
        "  [$primary]⌘V / ctrl+v[/]         attach the clipboard's image or file "
        "(text and paths paste as usual)\n"
        "  [$primary]⌘C / ctrl+c[/]         copy the selection "
        "[$text-muted](drag over the log to select)[/]\n"
        "  [$primary]Esc[/]                 close the menu · stop the running turn\n"
        "  [$primary]ctrl+q[/]              leave"
    )
    return Panel("commands", Content("\n").join(rows) + keys)


def status_panel(
    *,
    config: Config,
    ollama_host: str,
    installed: Sequence[ModelInfo] | None,
    mcp: str,
    mcp_file: str,
    skills: str,
    skills_dir: str,
    session_title: str,
    messages: int,
    home: str,
) -> Panel:
    """`/status`: the model, the keys, the local server, what is mounted,
    the session, where things are kept. `installed` is None when Ollama
    did not answer."""
    full = config.model if config.provider == "ollama" else ""
    lines: list[Content] = [
        Content.from_markup(
            "  [$text-muted]model[/]     $model[$text-muted]$full[/]",
            model=model_label(config),
            full=f"  {full}" if full and full != alias(full) else "",
        )
    ]
    for provider in KEYED:
        has_key = bool(config.key_for(provider))
        lines.append(
            Content.from_markup(
                "  [$text-muted]$label[/] $state",
                label=f"{PROVIDER_LABELS[provider]} key".ljust(9),
                state="set" if has_key else "not set",
            )
        )
    lines.append(
        Content.from_markup(
            "  [$text-muted]Ollama[/]    $host [$text-muted]· $state[/]",
            host=ollama_host,
            state=(
                "not answering"
                if installed is None
                else f"{len(installed)} model{'s' if len(installed) != 1 else ''} installed"
            ),
        )
    )
    lines.append(
        Content.from_markup(
            "  [$text-muted]mcp[/]       $state [$text-muted]· $file[/]", state=mcp, file=mcp_file
        )
    )
    lines.append(
        Content.from_markup(
            "  [$text-muted]skills[/]    $state [$text-muted]· $dir[/]",
            state=skills,
            dir=skills_dir,
        )
    )
    lines.append(
        Content.from_markup(
            "  [$text-muted]session[/]   $title [$text-muted]· $count messages[/]",
            title=session_title,
            count=messages,
        )
    )
    lines.append(Content.from_markup("  [$text-muted]home[/]      $home", home=home))
    return Panel("status", Content("\n").join(lines))
