"""Reading the account back out of a parts array: the `data-usage` parts
the accumulator kept, as `Usage` values, and their sum. The wire keys
are the protocol's — written in `events/wire.py`, read only here."""

from __future__ import annotations

from typing import Any, cast

from void_agent.core.usage import NO_USAGE, Usage


def usage_of(part: dict[str, Any]) -> Usage | None:
    """The `Usage` a `data-usage` part carries; None for any other part or
    a malformed one."""
    if part.get("type") != "data-usage":
        return None
    data = part.get("data")
    if not isinstance(data, dict):
        return None
    fields = cast("dict[str, Any]", data)
    try:
        return Usage(
            input=int(fields.get("input", 0)),
            output=int(fields.get("output", 0)),
            cache_read=int(fields.get("cacheRead", 0)),
            cache_write=int(fields.get("cacheWrite", 0)),
        )
    except (TypeError, ValueError):
        return None


def usages(parts: list[dict[str, Any]]) -> list[Usage]:
    """Every round-trip's usage in `parts`, in order."""
    return [usage for part in parts if (usage := usage_of(part)) is not None]


def total_usage(parts: list[dict[str, Any]]) -> Usage:
    """What every round-trip in `parts` cost together — zero when nothing
    was reported."""
    total = NO_USAGE
    for usage in usages(parts):
        total = total + usage
    return total
