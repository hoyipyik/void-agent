"""Short texts several widgets and the app say the same way: a path with
the home directory as `~`, the model as the status line names it, a
token count as people read one."""

from __future__ import annotations

from pathlib import Path

from cli.config import Config
from cli.providers.catalog import PROVIDER_LABELS
from cli.providers.ollama import alias
from void_agent import Usage


def tilde(path: Path) -> str:
    """A path with the home directory written as `~`."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def model_label(config: Config) -> str:
    """The provider and the model as the status line says them — an
    Ollama model by its alias."""
    if config.provider is None:
        return "no model"
    model = alias(config.model) if config.provider == "ollama" else config.model
    return f"{PROVIDER_LABELS[config.provider]} · {model}"


def bar_label(config: Config, agent: str, context: int = 0, consumed: int = 0) -> str:
    """What the status line's right side says: the agent, the model, the
    context the model read last, and what the session has consumed so far
    (in and out together) — each once a round-trip has said."""
    label = f"{agent} · {model_label(config)}"
    if context:
        label += f" · {tokens(context)} ctx"
    if consumed:
        label += f" · {tokens(consumed)} consumed"
    return label


def tokens(count: int) -> str:
    """A token count as people read one: `340`, `1.2k`, `12k`, `1.2M`."""
    if count < 1_000:
        return str(count)
    if count < 1_000_000:
        return _short(count / 1_000, "k")
    return _short(count / 1_000_000, "M")


def _short(value: float, unit: str) -> str:
    text = f"{value:.1f}" if value < 10 else f"{value:.0f}"
    return f"{text.removesuffix('.0')}{unit}"


def usage_label(usage: Usage) -> str:
    """One round-trip's cost in a line: the prompt, the answer, and how
    much of the prompt the cache served, when any."""
    line = f"{tokens(usage.input)} in · {tokens(usage.output)} out"
    if usage.cache_read:
        line += f" · {tokens(usage.cache_read)} cached"
    return line


def duration(seconds: float) -> str:
    """A span as people say one: `4s`, `1m 12s`, `1h 2m`."""
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, rest = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {rest}s" if rest else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"
