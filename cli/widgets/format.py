"""A part's payload as one line, or laid out in full: what the chips,
the cards and the ask card show of a tool's input and output."""

from __future__ import annotations

import json
from typing import Any, cast

INLINE_LIMIT = 96


def pretty(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def inline(value: Any, limit: int = INLINE_LIMIT) -> str:
    """One line of a value: a dict as `key: value, …`, anything else as
    compact JSON; cut at `limit` with an ellipsis."""
    if isinstance(value, dict):
        items = cast("dict[str, Any]", value)
        text = ", ".join(
            f"{key}: {json.dumps(item, ensure_ascii=False, default=str)}"
            for key, item in items.items()
        )
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def data_of(part: dict[str, Any]) -> dict[str, Any]:
    data = part.get("data")
    return cast("dict[str, Any]", data) if isinstance(data, dict) else {}
