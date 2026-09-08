"""The question card: the model's `ask_user`, or a gate holding a side
effect. Which key means "yes" is decided here — a signature card answers
with a boolean, every other card in words — never in core."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message as UiMessage
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from cli.widgets.format import inline, pretty

DIGITS = frozenset("123456789")


class AskCard(Vertical):
    """A question for the person — the model's `ask_user` or a gate holding
    a side effect. A card with a `call` asks for a signature and shows the
    exact call that runs when approved: what is shown is what is signed.
    Its rows answer with a boolean; a choice card's rows answer in words,
    its last row opening the composer for words of one's own; an input
    card is answered in the composer. The options are a list: the arrow
    keys move, Enter chooses, a digit jumps — the card holds the keys
    while it is open."""

    DEFAULT_CSS = """
    AskCard { border: round $primary; padding: 0 1; margin: 0 0 1 0; height: auto; }
    AskCard .ask-title { color: $primary; text-style: bold; }
    AskCard .ask-question { height: auto; }
    AskCard .ask-call { color: $text-muted; margin: 1 0 0 2; height: auto; }
    AskCard OptionList {
        height: auto; margin: 1 0 0 0; padding: 0; border: none; background: transparent;
    }
    AskCard OptionList:focus { border: none; }
    AskCard OptionList > .option-list--option { padding: 0 1; }
    AskCard OptionList > .option-list--option-highlighted {
        background: $block-cursor-background; color: $block-cursor-foreground; text-style: none;
    }
    AskCard OptionList.-answered { display: none; }
    AskCard .ask-status { display: none; color: $text-muted; margin: 1 0 0 0; height: auto; }
    AskCard .ask-hint { display: none; color: $text-muted; margin: 1 0 0 0; height: auto; }
    AskCard .ask-status.-shown, AskCard .ask-hint.-shown { display: block; }
    """
    SIGNATURE_OPTIONS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("approve", "Yes, run it"),
        ("decline", "No, don't"),
    )
    # The last row of a choice card: none of these — say it in words.
    OTHER = "__other__"

    class Answered(UiMessage):
        """The person chose on the card."""

        def __init__(self, ask_id: str, value: bool | str) -> None:
            super().__init__()
            self.ask_id = ask_id
            self.value = value

    class Words(UiMessage):
        """The person wants to answer this card in words: open the composer."""

        def __init__(self, ask_id: str) -> None:
            super().__init__()
            self.ask_id = ask_id

    def __init__(self, data: dict[str, Any], *, live: bool = True) -> None:
        super().__init__(classes="ask")
        self.ask_id = str(data.get("askId", ""))
        self.kind = str(data.get("kind", "input"))
        self.question = str(data.get("question", ""))
        self.options = [str(option) for option in cast("list[Any]", data.get("options") or [])]
        self.payload = data.get("payload")
        call = data.get("call")
        self.call: dict[str, Any] | None = (
            cast("dict[str, Any]", call) if isinstance(call, dict) else None
        )
        self.signature = self.call is not None
        self.answered: bool | str | None = None
        self.dropped = bool(data.get("dropped"))
        self._live = live
        self._choices = OptionList(*self._rows(), id="ask-options")
        self._status = Static("", classes="ask-status")
        self._hint = Static("", classes="ask-hint")

    def _rows(self) -> list[Option]:
        if self.signature or self.kind == "approval":
            pairs = list(self.SIGNATURE_OPTIONS)
        elif self.kind == "choice":
            pairs = [(option, option) for option in self.options]
            pairs.append((self.OTHER, "Other — say it below"))
        else:
            return []
        return [
            Option(Content.from_markup("$n. $label", n=index + 1, label=label), id=option_id)
            for index, (option_id, label) in enumerate(pairs)
        ]

    def compose(self) -> ComposeResult:
        if self.signature:
            yield Static("Approval needed", classes="ask-title")
        else:
            yield Static("Question", classes="ask-title")
        yield Static(Content(self.question), classes="ask-question")
        if self.call is not None:
            # What is shown is what is signed: the whole input, on one
            # line when it fits, laid out in full when it does not.
            args = inline(self.call.get("input", {}))
            shown = Content.from_markup(
                "[b]$tool[/]($args)", tool=str(self.call.get("tool", "?")), args=args
            )
            if args.endswith("…"):
                shown = shown + Content("\n" + pretty(self.call.get("input")))
            yield Static(shown, classes="ask-call")
        elif self.payload is not None:
            yield Static(Content(pretty(self.payload)), classes="ask-call")
        yield self._choices
        yield self._hint
        yield self._status

    def on_mount(self) -> None:
        self._show_state()
        if self.wants_focus() and self._choices.option_count:
            self._choices.highlighted = 0
            self._choices.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if self.answered is not None or self.dropped:
            return
        chosen = event.option.id or ""
        if chosen == self.OTHER:
            self._choices.add_class("-answered")
            self._say(
                self._hint,
                Content.from_markup(
                    "[$text-muted]type your answer in the prompt below and press Enter[/]"
                ),
            )
            self.post_message(self.Words(self.ask_id))
            return
        value: bool | str = chosen == "approve" if self.signature else chosen
        self.post_message(self.Answered(self.ask_id, value))

    def on_key(self, event: events.Key) -> None:
        """A digit jumps to that option and chooses it."""
        if event.key not in DIGITS or self._choices.disabled:
            return
        index = int(event.key) - 1
        if index >= self._choices.option_count:
            return
        event.stop()
        self._choices.highlighted = index
        self._choices.action_select()

    def update(self, data: dict[str, Any]) -> None:
        if data.get("dropped") and not self.dropped:
            self.dropped = True
            self._show_state()

    def mark_answered(self, value: bool | str) -> None:
        self.answered = value
        self._show_state()

    def finish(self) -> None:
        """The turn is over: whatever is still open cannot be answered here."""
        self._choices.disabled = True
        if self.answered is None and not self.dropped:
            self._say(self._hint, None)

    @staticmethod
    def _say(line: Static, content: Content | None) -> None:
        """A line that is there only while it has something to say."""
        line.update(content or "")
        line.set_class(content is not None, "-shown")

    def _show_state(self) -> None:
        if self.answered is not None:
            if isinstance(self.answered, bool):
                shown = "approved" if self.answered else "declined"
            else:
                shown = self.answered
            self._say(self._status, Content.from_markup("  ⎿  [$primary]$shown[/]", shown=shown))
            self._say(self._hint, None)
            self._choices.add_class("-answered")
            self.finish()
        elif self.dropped:
            self._say(
                self._status,
                Content.from_markup(
                    "  ⎿  [$text-muted]no answer came; the turn ended with the card open[/]"
                ),
            )
            self._say(self._hint, None)
            self._choices.add_class("-answered")
            self.finish()
        elif self._choices.option_count:
            self._say(
                self._hint,
                Content.from_markup("[$text-muted]↑↓ move · Enter choose · 1-9 jump[/]"),
            )
        else:
            self._say(
                self._hint,
                Content.from_markup(
                    "[$text-muted]type your answer in the prompt below and press Enter[/]"
                ),
            )

    def wants_focus(self) -> bool:
        return self._live and self.answered is None and not self.dropped
