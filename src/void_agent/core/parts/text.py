"""The plain-text projection of a parts array: titles and search. Only
text parts speak here; the model-facing rendering is `context_text`."""

from __future__ import annotations

from typing import Any


def parts_text(parts: list[dict[str, Any]]) -> str:
    """The plain text of a persisted parts array (titles, search). Only text
    parts speak here; for the model-facing rendering use `context_text`."""
    return "".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
