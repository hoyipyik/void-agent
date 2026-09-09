"""`/model`: the catalogue, grouped by cloud provider, then what Ollama
has installed that an agent can run on — where there is nothing, no
Ollama at all."""

from __future__ import annotations

from collections.abc import Sequence

from textual.content import Content
from textual.widgets.option_list import Option

from cli.config import Config
from cli.providers.catalog import (
    KEYED,
    PROVIDER_LABELS,
    ModelInfo,
    Provider,
    models_for,
    provider_of,
)
from cli.screens.chooser import DIALOG_CSS, Chooser


class ModelPicker(Chooser[tuple[Provider, str] | None]):
    """The catalogue, grouped by cloud provider, then what Ollama has
    installed that can call tools — read when the picker opens; empty
    when its server did not answer or has no such model, and then there
    is no Ollama section at all — the model in use marked. Dismisses
    with (provider, model id) or None."""

    # Wider than the other dialogs: an Ollama name carries its namespace and tag.
    CSS = DIALOG_CSS + "\n#dialog { width: 100; max-width: 100%; }"
    TITLE_TEXT = "Select a model"

    def __init__(self, config: Config, host: str, installed: Sequence[ModelInfo]) -> None:
        super().__init__()
        self._config = config
        self._host = host
        self._installed = tuple(installed)
        self.BLURB = (
            "Applies to this session and the ones after it. A cloud provider without a key"
            " asks for one next"
            + ("; an Ollama model needs none" if self._installed else "")
            + ". /model <id> names one not listed."
        )

    def _row(self, number: int, model: ModelInfo, width: int) -> Option:
        current = self._config.provider == model.provider and self._config.model == model.id
        return Option(
            Content.from_markup(
                "$mark $n $name [$text-muted]$blurb$star[/]",
                mark="●" if current else " ",
                n=f"{number}.".rjust(3) if number <= 9 else "   ",
                name=model.name.ljust(width),
                blurb=model.blurb,
                star="  · recommended" if model.recommended else "",
            ),
            id=model.id,
        )

    def rows(self) -> list[Option]:
        rows: list[Option] = []
        number = 0
        for provider in KEYED:
            rows.append(Option(PROVIDER_LABELS[provider], id=f"head-{provider}", disabled=True))
            for model in models_for(provider):
                number += 1
                rows.append(self._row(number, model, 14))
        if self._installed:
            rows.append(
                Option(
                    Content.from_markup(
                        "$label [$text-muted]· $host · $count usable[/]",
                        label=PROVIDER_LABELS["ollama"],
                        host=self._host,
                        count=len(self._installed),
                    ),
                    id="head-ollama",
                    disabled=True,
                )
            )
            width = min(max(len(model.name) for model in self._installed), 52)
            for model in self._installed:
                number += 1
                rows.append(self._row(number, model, width))
        return rows

    def initial(self) -> str | None:
        return self._config.model or None

    def chosen(self, option_id: str) -> tuple[Provider, str] | None:
        if any(model.id == option_id for model in self._installed):
            return ("ollama", option_id)
        provider = provider_of(option_id) or self._config.provider or "anthropic"
        return (provider, option_id)
