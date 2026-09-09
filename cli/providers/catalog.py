"""The models on offer: a curated catalogue per cloud provider — what
`/model` lists, with the name and the one line a person reads to choose.
The id is what the provider is asked for. The catalogue is a menu, not a
fence: `/model <id>` still reaches a model it does not list. Ollama, the
third provider, has no catalogue: its list is whatever the local server
has installed, read when the picker opens (`ollama.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["anthropic", "openai", "ollama"]
PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "ollama")
PROVIDER_LABELS: dict[Provider, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
}
# The providers that take a key; Ollama is a local server and takes none.
KEYED: tuple[Provider, ...] = ("openai", "anthropic")
KEY_VARIABLES: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def as_provider(value: object) -> Provider | None:
    return value if value in PROVIDERS else None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    provider: Provider
    id: str
    name: str
    blurb: str
    recommended: bool = False
    # Whether the model can call tools — what an agent runs on. Every
    # cloud model can; an Ollama model says so through `/api/show`.
    tools: bool = True


# The recommended model of each keyed provider is its default and heads
# its group. OpenAI comes first: Terra is what the CLI starts on.
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(
        "openai", "gpt-5.6-terra", "GPT-5.6 Terra", "intelligence and cost in balance", True
    ),
    ModelInfo("openai", "gpt-5.6-sol", "GPT-5.6 Sol", "complex professional work"),
    ModelInfo("openai", "gpt-6-astra", "GPT-6 Astra", "the most capable: the hardest work"),
    ModelInfo("openai", "gpt-5.6-luna", "GPT-5.6 Luna", "cost-sensitive, high-volume work"),
    ModelInfo("openai", "gpt-5.5", "GPT-5.5", "coding and professional work"),
    ModelInfo("openai", "gpt-5.4", "GPT-5.4", "the previous generation"),
    ModelInfo("openai", "gpt-5.4-mini", "GPT-5.4 mini", "small and quick"),
    ModelInfo("openai", "gpt-5.4-nano", "GPT-5.4 nano", "the cheapest: simple, high-volume tasks"),
    ModelInfo("anthropic", "claude-opus-5", "Opus 5", "complex reasoning and coding", True),
    ModelInfo("anthropic", "claude-sonnet-5", "Sonnet 5", "fast and balanced, for everyday work"),
    ModelInfo("anthropic", "claude-haiku-4-5", "Haiku 4.5", "the quickest and the cheapest"),
    ModelInfo("anthropic", "claude-fable-5-1", "Fable 5.1", "the most capable: long, hard tasks"),
    ModelInfo("anthropic", "claude-opus-4-8", "Opus 4.8", "the previous Opus"),
    ModelInfo("anthropic", "claude-opus-4-7", "Opus 4.7", "an earlier Opus"),
    ModelInfo("anthropic", "claude-opus-4-6", "Opus 4.6", "an earlier Opus"),
    ModelInfo("anthropic", "claude-sonnet-4-6", "Sonnet 4.6", "the previous Sonnet"),
)

DEFAULT_MODELS: dict[Provider, str] = {m.provider: m.id for m in CATALOG if m.recommended}


def models_for(provider: Provider) -> tuple[ModelInfo, ...]:
    return tuple(model for model in CATALOG if model.provider == provider)


def describe(model_id: str) -> ModelInfo | None:
    return next((model for model in CATALOG if model.id == model_id), None)


def provider_of(model_id: str) -> Provider | None:
    """The provider a model id names: the catalogue's word, else the
    family's — `claude-…` is Anthropic's, `gpt-…` and `o…` OpenAI's, a
    name with a tag (`qwen3:8b`) or a namespace (`hf.co/…`) Ollama's."""
    known = describe(model_id)
    if known is not None:
        return known.provider
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if ":" in model_id or "/" in model_id:
        return "ollama"
    return None
