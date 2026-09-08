"""Short texts several widgets and the app say the same way: a path with
the home directory as `~`, the model as the status line names it."""

from __future__ import annotations

from pathlib import Path

from cli.config import Config
from cli.providers.catalog import PROVIDER_LABELS
from cli.providers.ollama import alias


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


def bar_label(config: Config, agent: str) -> str:
    """What the status line's right side says: the agent, then the model."""
    return f"{agent} · {model_label(config)}"
