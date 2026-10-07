"""A tool's fingerprint: the bridge computes, from what a server published,
the value that server's own script wrote for the same tool. The examples in
`mcp_fingerprint_vectors.json` are shared, byte for byte, with the
publishers that follow the same rule: one that changes here changes there."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from void_agent.mcp.fingerprint import text_fingerprint, tool_fingerprint, tool_line

VECTORS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "mcp_fingerprint_vectors.json").read_text(encoding="utf-8")
)


def published(entry: dict[str, Any]) -> dict[str, Any]:
    """One `tools/list` entry as the five things a fingerprint reads."""
    annotations: dict[str, Any] = entry.get("annotations") or {}
    return {
        "name": entry["name"],
        "description": entry.get("description"),
        "input_schema": entry["inputSchema"],
        "read_only": annotations.get("readOnlyHint"),
        "destructive": annotations.get("destructiveHint"),
    }


PING: dict[str, Any] = {
    "name": "ping",
    "description": "Answers pong.",
    "input_schema": {"type": "object", "properties": {}},
    "read_only": True,
    "destructive": False,
}


@pytest.mark.parametrize("example", VECTORS["tools"], ids=lambda example: example["tool"]["name"])
def test_a_tool_gets_the_line_and_the_fingerprint_its_publisher_writes_for_it(
    example: dict[str, Any],
) -> None:
    tool = published(example["tool"])

    assert tool_line(**tool) == example["line"]
    assert tool_fingerprint(**tool) == example["fingerprint"]


def test_a_text_is_fingerprinted_as_the_hash_of_the_text_itself() -> None:
    instructions = VECTORS["instructions"]

    assert text_fingerprint(instructions["text"]) == instructions["fingerprint"]


@pytest.mark.parametrize(
    "changed",
    [
        {"name": "pong"},
        {"description": "Answers pong, promptly."},
        {"input_schema": {"type": "object", "properties": {"loud": {"type": "boolean"}}}},
        {"read_only": False},
        {"destructive": True},
    ],
    ids=lambda changed: next(iter(changed)),
)
def test_the_fingerprint_changes_with_anything_a_model_reads_or_a_mount_gates_on(
    changed: dict[str, Any],
) -> None:
    assert tool_fingerprint(**{**PING, **changed}) != tool_fingerprint(**PING)


def test_the_fingerprint_does_not_change_with_the_order_a_schema_was_written_in() -> None:
    reordered: dict[str, Any] = {**PING, "input_schema": {"properties": {}, "type": "object"}}

    assert tool_fingerprint(**reordered) == tool_fingerprint(**PING)


def test_a_hint_the_server_did_not_send_is_not_the_same_as_one_it_denied() -> None:
    unsaid: dict[str, Any] = {**PING, "read_only": None}
    denied: dict[str, Any] = {**PING, "read_only": False}

    assert tool_fingerprint(**unsaid) != tool_fingerprint(**denied)


def test_a_number_json_cannot_hold_is_refused() -> None:
    schema = {"type": "object", "maxProperties": float("inf")}
    unbounded: dict[str, Any] = {**PING, "input_schema": schema}

    with pytest.raises(ValueError, match="inf"):
        tool_fingerprint(**unbounded)
