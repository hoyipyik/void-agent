"""`/model`: the catalogue, grouped by cloud provider, then what Ollama
has installed that an agent can run on — where there is nothing, no
Ollama at all. Left and right set the effort of the model under the
cursor, along the levels it takes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from textual.binding import BindingType
from textual.content import Content
from textual.widgets.option_list import Option

from cli.config import Config
from cli.providers.catalog import (
    CATALOG,
    KEYED,
    PROVIDER_LABELS,
    ModelInfo,
    Provider,
    describe,
    models_for,
    provider_of,
)
from cli.screens.chooser import DIALOG_CSS, Chooser

CLOUD_NAME_WIDTH = 14
EFFORT_WIDTH = max(len(level) for model in CATALOG for level in model.efforts)
# What a row says before its effort: the mark, the number, the name.
EFFORT_COLUMN = 1 + 1 + 3 + 1 + CLOUD_NAME_WIDTH + 1

# (provider, model id, the effort the arrows set — None where they set none)
Picked = tuple[Provider, str, str | None]


class ModelPicker(Chooser[Picked | None]):
    """The catalogue, grouped by cloud provider, then what Ollama has
    installed that can call tools — read when the picker opens; empty
    when its server did not answer or has no such model, and then there
    is no Ollama section at all — the model in use marked. A cloud row
    shows the effort its model runs at — the provider's, where it takes
    it, else its own default — and the arrows move it. Dismisses with
    (provider, model id, effort) or None; Escape keeps no effort moved."""

    # Wider than the other dialogs: an Ollama name carries its namespace and tag.
    CSS = DIALOG_CSS + "\n#dialog { width: 100; max-width: 100%; }"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("left", "effort(-1)", "Less effort"),
        ("right", "effort(1)", "More effort"),
    ]
    TITLE_TEXT = "Select a model"
    HINT = "↑↓ move · ←→ effort · Enter choose · 1-9 jump · Esc cancel"

    def __init__(self, config: Config, host: str, installed: Sequence[ModelInfo]) -> None:
        super().__init__()
        self._config = config
        self._host = host
        self._installed = tuple(installed)
        self._efforts = {
            model.id: model.effort_at(config.effort_for(model.provider))
            for model in CATALOG
            if model.efforts
        }
        self._moved: set[str] = set()
        self._layout: dict[str, tuple[int, int, bool]] = {}
        self.BLURB = (
            "Applies to this session and the ones after it. A cloud provider without a key"
            " asks for one next"
            + ("; an Ollama model needs none" if self._installed else "")
            + ". /model <id> names one not listed."
        )

    def _prompt(self, number: int, model: ModelInfo, width: int, efforts: bool) -> Content:
        current = self._config.provider == model.provider and self._config.model == model.id
        return Content.from_markup(
            "$mark $n $name $effort[$text-muted]$blurb$star[/]",
            mark="●" if current else " ",
            n=f"{number}.".rjust(3) if number <= 9 else "   ",
            name=model.name.ljust(width),
            effort=Content.from_markup(
                "[$accent]$level[/] ",
                level=self._efforts.get(model.id, "").ljust(EFFORT_WIDTH),
            )
            if efforts
            else "",
            blurb=model.blurb,
            star="  · recommended" if model.recommended else "",
        )

    def _row(self, number: int, model: ModelInfo, width: int, efforts: bool = False) -> Option:
        self._layout[model.id] = (number, width, efforts)
        return Option(self._prompt(number, model, width, efforts), id=model.id)

    def rows(self) -> list[Option]:
        rows: list[Option] = []
        number = 0
        for provider in KEYED:
            head = Content.from_markup(
                "$label[$text-muted]effort[/]",
                label=PROVIDER_LABELS[provider].ljust(EFFORT_COLUMN),
            )
            rows.append(Option(head, id=f"head-{provider}", disabled=True))
            for model in models_for(provider):
                number += 1
                rows.append(self._row(number, model, CLOUD_NAME_WIDTH, efforts=True))
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

    def action_effort(self, step: int) -> None:
        """One level up or down the scale of the model under the cursor,
        stopping at its ends; a model with no scale ignores the arrows."""
        index = self.choices.highlighted
        if index is None:
            return
        model = describe(self.choices.get_option_at_index(index).id or "")
        if model is None or model.id not in self._efforts:
            return
        at = model.efforts.index(self._efforts[model.id])
        effort = model.efforts[max(0, min(len(model.efforts) - 1, at + step))]
        if effort == self._efforts[model.id]:
            return
        self._efforts[model.id] = effort
        self._moved.add(model.id)
        number, width, efforts = self._layout[model.id]
        self.choices.replace_option_prompt(model.id, self._prompt(number, model, width, efforts))

    def chosen(self, option_id: str) -> Picked | None:
        if any(model.id == option_id for model in self._installed):
            return ("ollama", option_id, None)
        provider = provider_of(option_id) or self._config.provider or "anthropic"
        return (
            provider,
            option_id,
            self._efforts[option_id] if option_id in self._moved else None,
        )
