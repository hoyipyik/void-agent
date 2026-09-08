"""The protocol, rendered: one `TurnView` per assistant message, its
widgets a function of the message's `parts` — the array the
`PartsAccumulator` folds live and the store keeps.

`sync(parts)` walks the array and mounts a widget for every part it has
not seen, updating the ones it has: a tool part changes state in place,
an ask card learns its answer from the `data-answer` part after it and
its fate from the `dropped` flag. A live turn syncs after every event and
streams text through the Markdown widget's own stream in between; a
reopened session syncs once over the stored array. One replay path.

Vocabulary: text → the reply, a dot in the
gutter; `dynamic-tool` → a line that folds open (`⏺ name(args)` and its
result under `⎿`); `data-plan` → a plan card where the update happened —
one per update, so the latest state is at the reading edge and the
earlier ones stay as the record of how it moved; `data-reflection` → a
card; `data-ask` → the question card, its options a list answered with
the keys; any other `data-*` → a folded card; `data-step` stays silent;
`data-error` and `data-cancelled` → a line. Which key means "yes" is
decided on the card — a signature card answers with a boolean, every
other card in words — never in core.
"""

from __future__ import annotations

from typing import Any, cast

from textual.containers import Vertical
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownStream

from cli.widgets.ask import AskCard
from cli.widgets.cards import PlanCard, ReflectionCard
from cli.widgets.fold import DataCard, ToolChip
from cli.widgets.format import data_of
from cli.widgets.reply import Reply, Said
from void_agent import AgentEvent, TextDelta, TextEnd, TextStart


class TurnView(Vertical):
    """One assistant message, rendered from its parts. `live` is a turn
    running now — its cards take the keys; a replayed one only shows."""

    DEFAULT_CSS = "TurnView { height: auto; }"

    def __init__(self, *, live: bool = True) -> None:
        super().__init__(classes="turn")
        self._live = live
        self._seen: dict[int, Widget | None] = {}
        self._asks: dict[str, AskCard] = {}
        self._streams: dict[str, MarkdownStream] = {}

    async def sync(self, parts: list[dict[str, Any]]) -> None:
        for part in parts:
            key = id(part)
            if key in self._seen:
                widget = self._seen[key]
                if isinstance(widget, ToolChip):
                    widget.update(part)
                elif isinstance(widget, AskCard):
                    widget.update(data_of(part))
                continue
            self._seen[key] = await self._mount_for(part)

    async def _mount_for(self, part: dict[str, Any]) -> Widget | None:
        kind = str(part.get("type", ""))
        widget: Widget | None = None
        match kind:
            case "text":
                reply = Reply(str(part.get("text", "")))
                await self.mount(Said(reply))
                return reply
            case "dynamic-tool":
                widget = ToolChip(part)
            case "data-plan":
                widget = PlanCard(cast("list[Any]", data_of(part).get("items") or []))
            case "data-reflection":
                widget = ReflectionCard(data_of(part))
            case "data-ask":
                card = AskCard(data_of(part), live=self._live)
                self._asks[card.ask_id] = card
                widget = card
            case "data-answer":
                data = data_of(part)
                card = self._asks.get(str(data.get("askId", "")))
                value = data.get("value")
                if card is not None and isinstance(value, bool | str):
                    card.mark_answered(value)
                return None
            case "data-step" | "data-ask-dropped":
                return None
            case "data-cancelled":
                widget = Static(Content.from_markup("[$text-muted]⏹ stopped[/]"), classes="note")
            case "data-error":
                widget = Static(
                    Content.from_markup(
                        "[$error]✗[/] $text", text=str(data_of(part).get("text", ""))
                    ),
                    classes="error",
                )
            case _ if kind.startswith("data-"):
                widget = DataCard(kind[len("data-") :], part.get("data"))
            case _:
                return None
        await self.mount(widget)
        return widget

    async def apply(self, event: AgentEvent, parts: list[dict[str, Any]]) -> None:
        """One live event: the parts changed, so sync; then stream text."""
        await self.sync(parts)
        match event:
            case TextStart(id=block_id):
                reply = self._seen.get(id(parts[-1])) if parts else None
                if isinstance(reply, Reply):
                    self._streams[block_id] = Markdown.get_stream(reply)
            case TextDelta(id=block_id, delta=delta):
                stream = self._streams.get(block_id)
                if stream is not None:
                    await stream.write(delta)
            case TextEnd(id=block_id):
                stream = self._streams.pop(block_id, None)
                if stream is not None:
                    await stream.stop()
            case _:
                pass

    async def finish(self) -> None:
        """The turn is over: flush the streams, close the open cards."""
        for stream in self._streams.values():
            await stream.stop()
        self._streams.clear()
        for card in self._asks.values():
            card.finish()

    def replies(self) -> list[Reply]:
        return list(self.query(Reply))
