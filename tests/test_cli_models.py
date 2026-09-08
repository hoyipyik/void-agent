"""The model catalogue `/model` lists."""

from __future__ import annotations

from cli.providers.catalog import (
    CATALOG,
    DEFAULT_MODELS,
    KEYED,
    PROVIDERS,
    describe,
    models_for,
    provider_of,
)


def test_every_cloud_provider_has_one_recommended_model_and_it_is_the_default() -> None:
    for provider in KEYED:
        recommended = [m for m in models_for(provider) if m.recommended]
        assert len(recommended) == 1
        assert DEFAULT_MODELS[provider] == recommended[0].id


def test_ollama_is_a_provider_without_a_catalogue_or_a_key() -> None:
    assert "ollama" in PROVIDERS
    assert "ollama" not in KEYED
    assert "ollama" not in DEFAULT_MODELS  # its models are whatever is installed
    assert models_for("ollama") == ()


def test_ids_are_unique_and_described() -> None:
    ids = [model.id for model in CATALOG]
    assert len(ids) == len(set(ids))
    assert all(model.name and model.blurb for model in CATALOG)
    assert describe("claude-opus-5") is not None
    assert describe("nope") is None


def test_a_model_id_names_its_provider_from_the_catalogue_or_its_family() -> None:
    assert provider_of("gpt-5.6-terra") == "openai"
    assert provider_of("claude-sonnet-5") == "anthropic"
    assert provider_of("claude-something-new") == "anthropic"
    assert provider_of("gpt-7") == "openai"
    assert provider_of("o4-mini") == "openai"
    assert provider_of("qwen3:8b") == "ollama"  # a tag is Ollama's naming
    assert provider_of("hf.co/someone/model-gguf:latest") == "ollama"  # so is a namespace
    assert provider_of("llama-3") is None
