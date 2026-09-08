"""The two builtin cards: the plan as it stood at one update, and the
model's self-assessment."""

from __future__ import annotations

from typing import Any, cast

from textual.content import Content
from textual.widgets import Static

PLAN_MARKS = {
    "completed": "[$success]☒[/] [$text-muted strike]$title[/]",
    "in_progress": "[$primary]▸[/] [b]$title[/]",
    "pending": "[$text-muted]☐[/] $title",
}


class PlanCard(Static):
    """The model's task list — its own progress narration, never system
    fact — as it stood at one update."""

    DEFAULT_CSS = "PlanCard { height: auto; margin: 0 0 1 0; }"

    def __init__(self, items: list[Any]) -> None:
        self.items: list[Any] = items
        super().__init__(self._render_items(items))

    @staticmethod
    def _render_items(items: list[Any]) -> Content:
        lines = [Content.from_markup("[$primary]⏺[/] [b]Plan[/]")]
        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            item = cast("dict[str, Any]", raw)
            status = str(item.get("status", "pending"))
            markup = PLAN_MARKS.get(status, PLAN_MARKS["pending"])
            gutter = "  ⎿  " if index == 0 else "     "
            line = Content(gutter) + Content.from_markup(markup, title=str(item.get("title", "")))
            if item.get("note"):
                line = line + Content.from_markup(
                    "  [$text-muted i]$note[/]", note=str(item["note"])
                )
            lines.append(line)
        return Content("\n").join(lines)


class ReflectionCard(Static):
    """The model's self-assessment: a verdict, the facts, the problems,
    the adjustment."""

    DEFAULT_CSS = "ReflectionCard { height: auto; margin: 0 0 1 0; }"

    def __init__(self, data: dict[str, Any]) -> None:
        self.verdict = str(data.get("verdict", "?"))
        lines = [
            Content.from_markup(
                "[$warning]⏺[/] [b]Reflection[/] [$text-muted]· $verdict[/]", verdict=self.verdict
            ),
            Content.from_markup("  ⎿  $facts", facts=str(data.get("facts", ""))),
        ]
        for problem in cast("list[Any]", data.get("problems") or []):
            lines.append(Content.from_markup("     [i]! $problem[/]", problem=str(problem)))
        if data.get("adjustment"):
            lines.append(Content.from_markup("     → $text", text=str(data["adjustment"])))
        super().__init__(Content("\n").join(lines))
