"""Ollama: the models a local (or named) Ollama server has installed, read
live — there is no catalogue to keep, the list is whatever `ollama pull`
put there. The model itself is spoken to by the OpenAI provider through
Ollama's OpenAI-compatible endpoint (`<host>/v1`, `cli/llm.py`), so
nothing here reaches core: this module only asks the server what it has
(`/api/tags`), and what each can do (`/api/show` — a model without
`tools` cannot drive an agent). Ollama is extra, never core: the CLI
looks for it itself, and offers it only where `usable` finds a model an
agent can run on."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import httpx

from cli.providers.catalog import ModelInfo

DEFAULT_PORT = 11434
DEFAULT_HOST = f"http://localhost:{DEFAULT_PORT}"
TIMEOUT = 3.0  # a local server answers at once; a named one may be away


class OllamaDown(Exception):
    """The server did not answer at `host`."""

    def __init__(self, host: str) -> None:
        super().__init__(f"Ollama is not answering at {host}")
        self.host = host


def host_url(value: str) -> str:
    """`OLLAMA_HOST` the way Ollama itself reads it — `localhost`,
    `0.0.0.0:11434`, `http://box:11434/` — as one base URL; empty is the
    default. A bare https host keeps its own port."""
    text = (value or "").strip().rstrip("/")
    if not text:
        return DEFAULT_HOST
    if "://" not in text:
        text = "http://" + text
    scheme, _, rest = text.partition("://")
    if ":" not in rest and scheme == "http":
        rest = f"{rest}:{DEFAULT_PORT}"
    return f"{scheme}://{rest}"


# The words a community build carries for "the refusals removed" — each
# becomes `U` in the alias; `gguf` is the file format, not a name.
_UNCENSORED = ("uncensored", "abliterated", "heretic")
# A tag that only says how the weights are quantised: the blurb says that
# already, so the alias drops it. One that says the size or the format
# (`8b`, `27b-mlx`, `122b-a10b`) is kept — it tells the models apart.
_QUANT_TAG = re.compile(r"^(latest|q\d.*|fp\d+|bf16|f16|f32|int\d+|nvfp4)$", re.IGNORECASE)


def alias(name: str, *, keep_tag: bool = False) -> str:
    """The short name the CLI shows and takes for an installed model: the
    namespace dropped (`orcarouter/`, `hf.co/…/`), a quantisation-only tag
    dropped, the base lowercased, an uncensored marker written `U` —
    `orcarouter/Qwen3.8-27B-Uncensored:q8_0` is `qwen3.8-27b-U`. The id
    sent to the server is always the full name."""
    base, _, tag = name.rsplit("/", 1)[-1].partition(":")
    kept: list[str] = []
    for word in base.lower().split("-"):
        if word == "gguf" or not word:
            continue
        if word in _UNCENSORED:
            if not kept or kept[-1] != "U":
                kept.append("U")
        else:
            kept.append(word)
    short = "-".join(kept) or base.lower()
    if not tag or tag == "latest" or (_QUANT_TAG.match(tag) and not keep_tag):
        return short
    return f"{short}:{tag}"


def _size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.1f} GB"
    return f"{size_bytes / 1e6:.0f} MB"


def _row(listed: dict[str, Any], shown: dict[str, Any]) -> ModelInfo:
    """One installed model as a row: the full name is the id, the alias the
    name; the blurb its parameters, quantisation and size, and `no tools`
    when it cannot call any."""
    name = str(listed.get("name") or listed.get("model") or "")
    details: dict[str, Any] = {}
    for source in (listed.get("details"), shown.get("details")):  # show's word wins
        if isinstance(source, dict):
            details.update({k: v for k, v in cast("dict[str, Any]", source).items() if v})
    bits = [
        str(details.get("parameter_size") or ""),
        str(details.get("quantization_level") or ""),
        _size(int(listed.get("size") or 0)) if listed.get("size") else "",
    ]
    capabilities = shown.get("capabilities")
    # A server that will not say what a model can do is given the benefit.
    tools = not isinstance(capabilities, list) or "tools" in cast("list[Any]", capabilities)
    if not tools:
        bits.append("no tools")
    blurb = " · ".join(bit for bit in bits if bit and bit != "unknown")
    return ModelInfo("ollama", name, alias(name), blurb, tools=tools)


def usable(installed: Sequence[ModelInfo]) -> tuple[ModelInfo, ...]:
    """The installed models an agent can run on — those that call tools.
    Empty, Ollama is not offered anywhere."""
    return tuple(model for model in installed if model.tools)


class Ollama:
    """One Ollama server, by its base URL; `transport` is the seam the
    tests answer through."""

    def __init__(
        self, host: str = DEFAULT_HOST, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.host = host
        self._transport = transport

    async def installed(self) -> tuple[ModelInfo, ...]:
        """Every installed model as a picker row. Raises `OllamaDown` when
        the server does not answer."""
        async with httpx.AsyncClient(
            base_url=self.host, transport=self._transport, timeout=TIMEOUT
        ) as client:
            try:
                response = await client.get("/api/tags")
                response.raise_for_status()
                tags = cast("dict[str, Any]", response.json())
                listed = [
                    cast("dict[str, Any]", entry)
                    for entry in cast("list[Any]", tags.get("models") or [])
                    if isinstance(entry, dict)
                ]
                shown = await asyncio.gather(*(self._show(client, entry) for entry in listed))
            except (httpx.HTTPError, ValueError) as error:
                raise OllamaDown(self.host) from error
        rows = [_row(entry, detail) for entry, detail in zip(listed, shown, strict=True)]
        # Two builds that differ only in their quantisation keep the tag.
        names = [row.name for row in rows]
        return tuple(
            replace(row, name=alias(row.id, keep_tag=True)) if names.count(row.name) > 1 else row
            for row in rows
        )

    async def _show(self, client: httpx.AsyncClient, entry: dict[str, Any]) -> dict[str, Any]:
        """What one model can do; nothing when the server will not say —
        the row is still listed."""
        name = str(entry.get("name") or entry.get("model") or "")
        try:
            response = await client.post("/api/show", json={"model": name})
            response.raise_for_status()
            detail = response.json()
        except (httpx.HTTPError, ValueError):
            return {}
        return cast("dict[str, Any]", detail) if isinstance(detail, dict) else {}


def find_installed(installed: Sequence[ModelInfo], name: str) -> str | None:
    """The installed model a bare name means — the full name, its alias, or
    `qwen3` for `qwen3:latest`, as Ollama itself takes it — else None."""
    return next(
        (
            model.id
            for model in installed
            if name in (model.id, model.id.removesuffix(":latest"))
            or name.lower() == model.name.lower()
        ),
        None,
    )
