"""`/key`, and the first thing a shell with no provider asks: a provider,
then its key."""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from cli.config import Config
from cli.providers.catalog import (
    DEFAULT_MODELS,
    KEY_VARIABLES,
    KEYED,
    PROVIDER_LABELS,
    PROVIDERS,
    Provider,
    as_provider,
)
from cli.screens.chooser import DIALOG_CSS, DIGITS


class KeyPrompt(ModalScreen[tuple[Provider, str] | None]):
    """A provider, then its key. The key is saved to the config file,
    private to the user. Dismisses with (provider, key) or None — for
    Ollama, which takes no key, with ("ollama", "") at once: the app
    opens the model picker instead."""

    CSS = DIALOG_CSS
    BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

    def __init__(
        self, config: Config, config_file_hint: str, provider: Provider | None = None
    ) -> None:
        super().__init__()
        self._provider: Provider | None = provider
        self._hint = config_file_hint

    def compose(self) -> ComposeResult:
        rows = [
            Option(
                Content.from_markup(
                    "$n. $label [$text-muted]$model · $variable[/]",
                    n=index + 1,
                    label=PROVIDER_LABELS[provider].ljust(10),
                    model=DEFAULT_MODELS[provider],
                    variable=KEY_VARIABLES[provider],
                ),
                id=provider,
            )
            if provider in KEYED
            else Option(
                Content.from_markup(
                    "$n. $label [$text-muted]no key — a model it has installed, OLLAMA_MODEL[/]",
                    n=index + 1,
                    label=PROVIDER_LABELS[provider].ljust(10),
                ),
                id=provider,
            )
            for index, provider in enumerate(PROVIDERS)
        ]
        with Vertical(id="dialog"):
            yield Static("Choose a provider", classes="title")
            yield Static(
                Content.from_markup(
                    "A cloud provider takes its key, saved to $path, readable only by you"
                    " (the environment variable beside it works too). Ollama takes none:"
                    " it lists what is installed.",
                    path=self._hint,
                ),
                classes="blurb",
            )
            yield OptionList(*rows, id="providers")
            yield Input(placeholder="sk-…", password=True, id="key")
            yield Static("", classes="hint")

    def on_mount(self) -> None:
        if self._provider is None:
            self.query_one("#key", Input).display = False
            self.query_one(".hint", Static).update(
                Content.from_markup(
                    "[$text-muted]↑↓ move · Enter choose · Esc go on without a model[/]"
                )
            )
            providers = self.query_one("#providers", OptionList)
            providers.highlighted = 0
            providers.focus()
        else:
            self._ask_for_key(self._provider)

    def _ask_for_key(self, provider: Provider) -> None:
        self._provider = provider
        providers = self.query_one("#providers", OptionList)
        providers.display = False
        key = self.query_one("#key", Input)
        key.placeholder = f"{PROVIDER_LABELS[provider]} API key"
        key.display = True
        self.query_one(".hint", Static).update(
            Content.from_markup(
                "[$text-muted]$label · Enter to save · Esc to go on without a model[/]",
                label=PROVIDER_LABELS[provider],
            )
        )
        key.focus()

    def _chose(self, provider: Provider) -> None:
        if provider == "ollama":
            self.dismiss(("ollama", ""))
        else:
            self._ask_for_key(provider)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        provider = as_provider(event.option.id)
        if provider is not None:
            self._chose(provider)

    def on_key(self, event: events.Key) -> None:
        if self._provider is not None or event.key not in DIGITS:
            return
        index = int(event.key) - 1
        if index < len(PROVIDERS):
            event.stop()
            self._chose(PROVIDERS[index])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        key = event.value.strip()
        if key and self._provider is not None:
            self.dismiss((self._provider, key))

    def action_cancel(self) -> None:
        self.dismiss(None)
