"""The shell around one session: the screen the app shows.

Everything here is the session's — the log, the composer and the menu
that unfolds above it, the rewind, the attachments waiting for the next
message, the turn and the questions it asks. What is the process's — the
config and the file it lives in, the mounted agents, servers and skills,
the model and its key — is the app's (`cli/app.py`), reached as `void`.

The runtime runs in-process: a turn is `agent.run(history, events,
human=…)` on the app's own event loop (`cli/session/runner.py`), its stream
drained the way a server drains it, each event folded into `parts` and
rendered by the turn's `TurnView` (`cli/widgets/turn.py`). The session
lives on disk (`cli/session/store.py`) as parts; what the next turn's model
reads is the session's own projection of them — semantic resume without
a server.

The person attends the run through a `HumanChannel`: every question —
the model's `ask_user`, a gate three layers down — arrives as a
`Question` and goes out as a card. A signature card's list replies with
a boolean, a choice card's with the option; an input card is answered in
the composer, which opens for it while the turn waits (`cli/session/asks.py`).
The reply wakes the frame that asked, in place. Escape stops a turn: the
run is cancelled, the message marked cancelled, nothing else changes.

Attachments (`cli/session/attachments.py`) ride the message as `file` parts: a
path pasted or dragged into the composer, an `@path` in the text, the OS
clipboard on ctrl+v or `/paste` (`cli/clipboard.py`), `/attach <path>`.
They wait in the bar above the prompt until the message goes.

Slash commands (`cli/commands.py`) unfold in a menu above the prompt as
"/" is typed: the arrows move, Tab completes, Enter runs. The session's
own — `/session`, `/new`, `/clear`, the attachments, `/status`, `/help`
— are run here; the rest are the app's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import Screen
from textual.widgets import OptionList, Static, TextArea

from cli.commands import (
    Attach,
    Clear,
    Command,
    Detach,
    Help,
    New,
    Paste,
    Quit,
    SessionPick,
    Status,
    Unknown,
    complete,
    matching,
    parse,
)
from cli.labels import bar_label
from cli.screens import SessionPicker
from cli.session import Session, SessionStore
from cli.session.asks import Desk
from cli.session.attachments import mentions, read_attachment
from cli.session.runner import Turn
from cli.widgets.ask import AskCard
from cli.widgets.attachbar import AttachmentBar
from cli.widgets.composer import Composer
from cli.widgets.menu import CommandMenu
from cli.widgets.panels import help_panel
from cli.widgets.prompt import REWIND_HINT, PromptFrame, StatusBar, activity
from cli.widgets.reply import Reply, UserBubble
from cli.widgets.turn import TurnView
from cli.widgets.welcome import Welcome
from void_agent import AgentEvent, Question, parts_text

if TYPE_CHECKING:
    from cli.app import VoidApp

PROMPT = "Ask void anything…  (/ for commands, @path to attach)"
ANSWER_PROMPT = "Answer the question above…"
# While a turn runs the box is closed, and says so.
BUSY_PROMPT = "Waiting for the reply…  (esc stops the turn)"
# How long a word flashed in the status line — "copied" — stays.
FLASH_SECONDS = 2.0


class Shell(Screen[None]):
    CSS = """
    #log { padding: 0 1; scrollbar-size: 1 1; }
    .note { color: $text-muted; margin: 0 0 1 0; height: auto; }
    .error { color: $text-error; margin: 0 0 1 0; height: auto; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [("escape", "stop", "Stop")]

    class Ready(Message):
        """The log is there: whatever the app has to say can go on it."""

    def __init__(self, store: SessionStore) -> None:
        super().__init__()
        self.store = store
        self.session: Session = store.new()
        # The live turn, if any, and the questions it is waiting on.
        self._inflight: Turn | None = None
        self.desk = Desk()
        # The user message being rewound to — its index in the session —
        # while ↑ has taken the composer back to it.
        self._rewind: int | None = None
        self._composer = Composer(PROMPT)
        self._menu = CommandMenu()
        self._prompt = PromptFrame(self._composer)
        self._status = StatusBar()
        self.attachments = AttachmentBar()

    @property
    def void(self) -> VoidApp:
        """The app, as what it is."""
        return cast("VoidApp", self.app)  # pyright: ignore[reportUnknownMemberType]

    def compose(self) -> ComposeResult:
        # The log takes no focus: a click or a drag on it — to fold a chip,
        # to select — leaves the keys where they were, on the composer or
        # an open card.
        yield VerticalScroll(id="log", can_focus=False)
        yield self.attachments
        yield self._menu
        yield self._prompt
        yield self._status

    async def on_mount(self) -> None:
        self.query_one("#log", VerticalScroll).anchor()
        await self.show_welcome()
        self.refresh_label()
        self.composer.focus()
        self.post_message(self.Ready())

    def refresh_label(self) -> None:
        """The agent and the model as the config stands now: the status
        line's right side, and the welcome box at the top of the log."""
        config, agent = self.void.config, self.void.agent_label()
        self._status.show_model(bar_label(config, agent))
        for welcome in self.query(Welcome):
            welcome.show(config, agent)

    def flash(self, text: str) -> None:
        """A word in the status line for a moment — "copied" — then what
        was there: the rewind's hint, or the idle one. While a turn runs
        the spinner keeps the line."""
        self._status.show_hint(text)
        self.set_timer(FLASH_SECONDS, self._unflash)

    def _unflash(self) -> None:
        if self._rewind is not None:
            self._status.show_hint(REWIND_HINT)
        elif self._inflight is None:
            self._status.idle()

    # ── what the tests and the person see ──────────────────────────────

    def replies(self) -> list[Reply]:
        return list(self.query(Reply))

    @property
    def composer(self) -> Composer:
        return self._composer

    @property
    def menu(self) -> CommandMenu:
        return self._menu

    @property
    def status(self) -> StatusBar:
        return self._status

    async def append(self, widget: Static | TurnView | Welcome) -> None:
        await self.query_one("#log", VerticalScroll).mount(widget)

    async def note(self, text: str) -> None:
        await self.append(
            Static(Content.from_markup("[$primary]⏺[/] $text", text=text), classes="note")
        )

    async def complain(self, text: str) -> None:
        await self.append(
            Static(Content.from_markup("[$error]✗[/] $text", text=text), classes="error")
        )

    async def clear_log(self) -> None:
        await self.query_one("#log", VerticalScroll).remove_children()

    async def show_welcome(self) -> None:
        app = self.void
        await self.append(Welcome(app.config, app.home, app.agent_label()))

    # ── the composer and the menu ──────────────────────────────────────

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self._composer:
            self._refresh_menu()

    def _refresh_menu(self) -> None:
        if self.desk.answering is not None:
            self._hide_menu()
            return
        specs = matching(self._composer.text)
        if specs:
            self._menu.show(specs)
            self._composer.menu_open = True
        else:
            self._hide_menu()

    def _hide_menu(self) -> None:
        self._menu.hide()
        self._composer.menu_open = False

    def on_composer_navigate(self, message: Composer.Navigate) -> None:
        key = message.key
        if self._menu.open:
            self._menu_key(key)
        elif self._rewind is not None:
            self._rewind_key(key)
        elif key == "up" and self.desk.answering is None:
            indexes = self.session.user_indexes()
            if indexes:
                self._rewind_to(indexes[-1])

    def _menu_key(self, key: str) -> None:
        match key:
            case "up" | "down":
                self._menu.move(key)
            case "tab":
                spec = self._menu.chosen
                if spec is not None:
                    completed = complete(self._composer.text, spec)
                    if spec.argument and completed == f"/{spec.name}":
                        completed += " "
                    self._composer.load(completed)
            case "escape":
                self._hide_menu()
            case _:
                pass

    # ── the rewind: ↑ takes the composer back to an earlier message ────

    def _rewind_key(self, key: str) -> None:
        indexes = self.session.user_indexes()
        if self._rewind not in indexes:
            self._exit_rewind()
            return
        position = indexes.index(self._rewind)
        match key:
            case "up":
                if position > 0:
                    self._rewind_to(indexes[position - 1])
            case "down":
                if position + 1 < len(indexes):
                    self._rewind_to(indexes[position + 1])
                else:
                    self._exit_rewind()
            case "escape":
                self._exit_rewind()
            case _:
                pass

    def _rewind_to(self, index: int) -> None:
        """Mark the user message at `index` and put its words in the
        composer: what is sent next replaces it and everything after."""
        self._rewind = index
        self._composer.rewinding = True
        self._composer.load(parts_text(self.session.messages[index].parts).strip())
        bubbles = list(self.query(UserBubble))
        for bubble, message_index in zip(bubbles, self.session.user_indexes(), strict=False):
            bubble.set_class(message_index == index, "-target")
            if message_index == index:
                bubble.scroll_visible()
        self._status.show_hint(REWIND_HINT)

    def _exit_rewind(self) -> None:
        if self._rewind is None:
            return
        self._rewind = None
        self._composer.rewinding = False
        self._composer.clear()
        for bubble in self.query(UserBubble):
            bubble.remove_class("-target")
        if self._inflight is None:
            self._status.idle()

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        text = message.text
        chosen = self._menu.chosen
        self._hide_menu()
        answering = self.desk.answering
        if answering is not None:
            self._answer(answering.ask.ask_id, text)
            return
        replacing = self._rewind
        self._exit_rewind()
        if text.startswith("/"):
            command = parse(complete(text, chosen) if chosen is not None else text)
            if command is not None:
                await self._run_command(command)
                return
        text, mentioned = mentions(text)
        for path in mentioned:
            await self._add_attachment(path)
        parts = self.attachments.parts()
        if text:
            parts.append({"type": "text", "text": text})
        if not parts:
            return
        self.attachments.drop()
        if replacing is not None:
            # The rewound message and everything after it go; the log is
            # drawn again from what is left, then the replacement is sent.
            self.session.truncate(replacing)
            await self._open(self.session)
        await self._send(parts)

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """A click on the command menu runs that command."""
        if event.option_list is not self._menu:
            return
        event.stop()
        spec = self._menu.spec_named(event.option.id)
        text = self._composer.text
        self._composer.clear()
        self._hide_menu()
        if spec is not None:
            command = parse(complete(text if text.startswith("/") else "/", spec))
            if command is not None:
                await self._run_command(command)
        self._composer.focus()

    # ── attachments ────────────────────────────────────────────────────

    @property
    def pending(self) -> list[Any]:
        return self.attachments.pending

    async def on_composer_pasted_paths(self, message: Composer.PastedPaths) -> None:
        for path in message.paths:
            await self._add_attachment(path)

    def on_composer_clipboard_requested(self, message: Composer.ClipboardRequested) -> None:
        self.run_worker(self._paste_clipboard())

    async def _add_attachment(self, path: Path) -> None:
        read = read_attachment(path)
        if isinstance(read, str):
            await self.complain(read)
            return
        self.attachments.add([read])

    async def _paste_clipboard(self) -> None:
        read = await asyncio.to_thread(self.void.os_clipboard.read)
        if isinstance(read, str):
            await self.complain(read)
            return
        self.attachments.add(read)

    async def _send(self, parts: list[dict[str, Any]]) -> None:
        await self.append(UserBubble(parts))
        view = TurnView()
        await self.append(view)
        self.session.append("user", parts)
        self.store.save(self.session)
        self.run_worker(self._turn(view))

    # ── commands: the session's here, the rest the app's ───────────────

    async def _run_command(self, command: Command) -> None:
        match command:
            case SessionPick():
                self.void.push_screen(
                    SessionPicker(self.store.list(), self.session.id), self._session_picked
                )
            case New():
                await self._open(self.store.new())
            case Clear():
                self.session.clear()
                self.store.save(self.session)
                await self.clear_log()
                await self.show_welcome()
                await self.note("session cleared")
            case Attach(path=path):
                if not path:
                    await self.complain("usage: /attach <path>")
                else:
                    await self._add_attachment(Path(path))
            case Paste():
                self.run_worker(self._paste_clipboard())
            case Detach():
                self.attachments.drop()
            case Status():
                await self.append(await self.void.status())
            case Help():
                await self.append(help_panel())
            case Quit():
                self.void.exit()
            case Unknown(name=name):
                await self.complain(f"unknown command /{name} — /help lists them")
            case _:
                await self.void.run_command(command)

    # ── sessions ───────────────────────────────────────────────────────

    async def _session_picked(self, choice: str | None) -> None:
        if choice is None:
            return
        if choice == SessionPicker.NEW:
            await self._open(self.store.new())
        else:
            await self.reopen(choice)

    async def reopen(self, session_id: str) -> None:
        await self._open(self.store.load(session_id))

    async def _open(self, session: Session) -> None:
        """Make `session` the current one and replay its log from parts —
        the same rendering a live turn leaves behind."""
        self._exit_rewind()
        self.session = session
        await self.clear_log()
        await self.show_welcome()
        for message in session.messages:
            if message.role == "user":
                await self.append(UserBubble(message.parts))
            else:
                view = TurnView(live=False)
                await self.append(view)
                await view.sync(message.parts)
                await view.finish()

    # ── the person answers ─────────────────────────────────────────────

    def on_ask_card_answered(self, message: AskCard.Answered) -> None:
        self._answer(message.ask_id, message.value)

    def _answer(self, ask_id: str, value: bool | str) -> None:
        """Reply to the question waiting under that card; if the composer
        was open for it, it closes."""
        was = self.desk.answering
        if self.desk.answer(ask_id, value) and was is not None and self.desk.answering is None:
            self._composer_waits(True)

    def _composer_waits(self, waiting: bool) -> None:
        composer = self.composer
        composer.disabled = waiting
        answering = not waiting and self.desk.answering is not None
        composer.placeholder = ANSWER_PROMPT if answering else BUSY_PROMPT if waiting else PROMPT
        self._prompt.set_sign("[$warning]?[/]" if answering else "❯")  # noqa: RUF001
        self._prompt.set_class(waiting, "-busy")
        if not waiting:
            composer.focus()

    def _question_arrived(self, question: Question) -> None:
        # Words are typed: the composer opens for the answer. Every other
        # card holds the keys itself, so nothing here takes its focus.
        if self.desk.arrive(question):
            self._composer_waits(False)

    def on_ask_card_words(self, message: AskCard.Words) -> None:
        if self.desk.words(message.ask_id):
            self._composer_waits(False)

    def action_stop(self) -> None:
        if self._inflight is not None:
            self._inflight.cancel()

    # ── a turn ─────────────────────────────────────────────────────────

    async def _turn(self, view: TurnView) -> None:
        # One turn at a time: the composer waits, as the web's Send does.
        self._composer_waits(True)
        self._status.busy("Thinking…")
        try:
            parts = await self._stream(view)
            if parts:
                self.session.append("assistant", parts)
                self.store.save(self.session)
        finally:
            await view.finish()
            self.desk.clear()
            self._status.idle()
            self._composer_waits(False)

    async def _stream(self, view: TurnView) -> list[dict[str, Any]]:
        """Run the turn (`cli/session/runner.py`), the view following every event
        and the status line saying what the turn is doing, and return the
        folded parts."""
        app = self.void
        turn = Turn(app.build_agent(app.config), self.session.history())
        self._inflight = turn

        async def on_event(event: AgentEvent, parts: list[dict[str, Any]]) -> None:
            said = activity(event)
            if said is not None:
                self._status.busy(said)
            await view.apply(event, parts)

        try:
            parts = await turn.drain(on_event, self._question_arrived)
        finally:
            self._inflight = None
        await view.sync(parts)
        return parts
