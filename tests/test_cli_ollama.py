"""What Ollama has installed, read from its server when the picker opens —
there is no catalogue to keep for a local model."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
from cli.providers.ollama import DEFAULT_HOST, Ollama, OllamaDown, alias, host_url, usable

TAGS: dict[str, Any] = {
    "models": [
        {
            "name": "qwen3:8b",
            "size": 5_225_000_000,
            "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M"},
        },
        {
            "name": "gemma4:31b-mlx",
            "size": 20_000_000_000,
            "details": {"parameter_size": "", "quantization_level": "nvfp4"},
        },
        {"name": "tiny:latest", "size": 400_000_000, "details": {}},
        {
            "name": "someone/Qwen3.8-27B-Uncensored:q8_0",
            "size": 30_000_000_000,
            "details": {"parameter_size": "27.3B", "quantization_level": "Q8_0"},
        },
    ]
}
SHOWS: dict[str, dict[str, Any]] = {
    "qwen3:8b": {"capabilities": ["completion", "tools", "thinking"]},
    "gemma4:31b-mlx": {
        "capabilities": ["completion", "vision", "tools"],
        "details": {"parameter_size": "31.3B", "quantization_level": "nvfp4"},
    },
    "tiny:latest": {"capabilities": ["completion"], "details": {"quantization_level": "unknown"}},
    "someone/Qwen3.8-27B-Uncensored:q8_0": {"capabilities": ["completion", "tools"]},
}


def server(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/tags":
        return httpx.Response(200, json=TAGS)
    if request.url.path == "/api/show":
        name = cast("dict[str, str]", json.loads(request.content))["model"]
        return httpx.Response(200, json=SHOWS[name])
    return httpx.Response(404)


def down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


async def test_the_installed_models_become_rows_with_their_size_and_what_they_can_do() -> None:
    ollama = Ollama("http://box:11434", transport=httpx.MockTransport(server))
    rows = await ollama.installed()
    assert [(row.provider, row.id, row.name) for row in rows] == [
        ("ollama", "qwen3:8b", "qwen3:8b"),
        ("ollama", "gemma4:31b-mlx", "gemma4:31b-mlx"),
        ("ollama", "tiny:latest", "tiny"),
        ("ollama", "someone/Qwen3.8-27B-Uncensored:q8_0", "qwen3.8-27b-U"),
    ]
    assert rows[0].blurb == "8.2B · Q4_K_M · 5.2 GB"
    assert rows[1].blurb == "31.3B · nvfp4 · 20.0 GB"  # /api/show fills in what /api/tags lacks
    assert rows[2].blurb == "400 MB · no tools"  # an agent cannot run on it


async def test_only_the_models_that_can_call_tools_are_usable() -> None:
    """Ollama is offered where it has a model an agent can run on; the
    rest are still installed — `/status` counts them — but never listed."""
    rows = await Ollama("http://box:11434", transport=httpx.MockTransport(server)).installed()
    assert [row.tools for row in rows] == [True, True, False, True]
    assert [row.id for row in usable(rows)] == [
        "qwen3:8b",
        "gemma4:31b-mlx",
        "someone/Qwen3.8-27B-Uncensored:q8_0",
    ]
    assert usable(()) == ()


async def test_a_server_that_does_not_answer_is_down() -> None:
    ollama = Ollama("http://box:11434", transport=httpx.MockTransport(down))
    with pytest.raises(OllamaDown) as caught:
        await ollama.installed()
    assert caught.value.host == "http://box:11434"


def test_the_alias_is_the_name_without_the_namespace_the_quantisation_or_euphemisms() -> None:
    assert alias("qwen3:8b") == "qwen3:8b"  # a size tag tells the builds apart
    assert alias("qwen3.8:27b-mlx") == "qwen3.8:27b-mlx"
    assert alias("phi4-mini:latest") == "phi4-mini"
    assert alias("orcarouter/Qwen3.8-27B-Uncensored:q8_0") == "qwen3.8-27b-U"
    assert alias("huihui_ai/qwen2.5-vl-abliterated:latest") == "qwen2.5-vl-U"
    assert alias("hf.co/Stabhappy/gemma-4-31B-it-heretic-Gguf:latest") == "gemma-4-31b-it-U"
    assert alias("Uncensored-Abliterated:latest") == "U"  # one marker, however many words
    assert alias("llama3:fp16") == "llama3"
    assert alias("llama3:fp16", keep_tag=True) == "llama3:fp16"


async def test_two_builds_that_differ_only_in_quantisation_keep_their_tags() -> None:
    tags = {
        "models": [
            {"name": "llama3:q4_K_M", "size": 1_000_000_000},
            {"name": "llama3:q8_0", "size": 2_000_000_000},
            {"name": "llama3:8b", "size": 2_000_000_000},
        ]
    }

    def twins(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        return httpx.Response(200, json={})

    rows = await Ollama("http://box:11434", transport=httpx.MockTransport(twins)).installed()
    assert [row.name for row in rows] == ["llama3:q4_K_M", "llama3:q8_0", "llama3:8b"]
    assert all(row.tools for row in rows)  # a server that will not say is given the benefit


def test_ollama_host_is_read_the_way_ollama_reads_it() -> None:
    assert host_url("") == DEFAULT_HOST
    assert host_url("localhost") == "http://localhost:11434"
    assert host_url("0.0.0.0:11434") == "http://0.0.0.0:11434"
    assert host_url("http://box:8080/") == "http://box:8080"
    assert host_url("https://box") == "https://box"
