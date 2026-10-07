"""A tool's fingerprint: the SHA-256 of what a server published for it.

Whoever mounts a server can pin a tool by this value and see, at the next
mount, that the server changed what a model reads or what a gate is decided
on. Five things are hashed: the tool's name, its description, its input
schema, and the server's read-only and destructive hints. What the server
did not send is null. Its title, its other hints and `_meta` are left out.

A publisher writes the same value from its own side, in whatever language
it is written in, so the line that is hashed is canonical JSON (RFC 8785):
keys in UTF-16 code unit order, arrays as given, no whitespace, strings and
numbers as JavaScript writes them. The fingerprint is `sha256:` and the hex
digest of that line's UTF-8 bytes. A text — a server's instructions — is
fingerprinted as the text itself.

Nothing here knows the MCP SDK: `server.py` reads the descriptor and hands
over the five values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from math import isfinite
from typing import Any, cast


def tool_line(
    *,
    name: str,
    description: str | None,
    input_schema: Mapping[str, Any],
    read_only: bool | None,
    destructive: bool | None,
) -> str:
    """What a model reads and what a mount gates on, as the one line that is
    hashed. What the server did not send is None."""
    return _canonical(
        {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
        }
    )


def tool_fingerprint(
    *,
    name: str,
    description: str | None,
    input_schema: Mapping[str, Any],
    read_only: bool | None,
    destructive: bool | None,
) -> str:
    return text_fingerprint(
        tool_line(
            name=name,
            description=description,
            input_schema=input_schema,
            read_only=read_only,
            destructive=destructive,
        )
    )


def text_fingerprint(text: str) -> str:
    """A server's instructions are fingerprinted as the text itself."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return _number(float(value))
    if isinstance(value, (list, tuple)):
        items = cast("list[object]", value)
        return "[" + ",".join(_canonical(item) for item in items) + "]"
    if isinstance(value, Mapping):
        members = cast("Mapping[str, object]", value)
        keys = sorted(members, key=lambda key: key.encode("utf-16-be"))
        written = (f"{_canonical(key)}:{_canonical(members[key])}" for key in keys)
        return "{" + ",".join(written) + "}"
    raise TypeError(f"a fingerprint holds JSON, not {type(value).__name__}")


def _number(value: float) -> str:
    """A number as JavaScript writes it: 100, not 100.0; 0.000001 but 1e-7;
    1e+21. Python's own repr differs in each of these."""
    if not isfinite(value):
        raise ValueError(f"a fingerprint cannot hold {value}: JSON has no such number")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # repr is the shortest text that reads back as this float, which is the
    # digits JavaScript prints; only where the point goes is decided here.
    _, digits, exponent = Decimal(repr(abs(value))).as_tuple()
    assert isinstance(exponent, int)
    text = "".join(map(str, digits)).rstrip("0")
    point = len(digits) + exponent
    if len(text) <= point <= 21:
        return sign + text + "0" * (point - len(text))
    if 0 < point <= 21:
        return sign + text[:point] + "." + text[point:]
    if -6 < point <= 0:
        return sign + "0." + "0" * -point + text
    mantissa = text[0] + ("." + text[1:] if len(text) > 1 else "")
    return f"{sign}{mantissa}e{'+' if point > 0 else '-'}{abs(point - 1)}"
