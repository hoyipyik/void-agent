"""The models on offer: a curated catalogue per cloud provider — what
`/model` lists, with the name and the one line a person reads to choose.
The id is what the provider is asked for. The catalogue is a menu, not a
fence: `/model <id>` still reaches a model it does not list. Ollama, the
third provider, has no catalogue: its list is whatever the local server
has installed, read when the picker opens (`ollama.py`).

A model's efforts are the reasoning levels its own API accepts, read off
it model by model — OpenAI's by asking each (2026-10-01), Anthropic's from
its documentation (2026-10-19) — since a level one model takes, its
neighbour may refuse with a 400. OpenAI's `none` — no reasoning at all —
is the least on the scale of a model that takes it: there for the person
to choose, never a default. What the CLI asks for is always on the scale
and never left to the API, whose own default for some models is that
`none`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["anthropic", "openai", "ollama"]
PROVIDERS: tuple[Provider, ...] = ("anthropic", "openai", "ollama")
PROVIDER_LABELS: dict[Provider, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
}
# The providers that take a key; Ollama is a local server and takes none.
KEYED: tuple[Provider, ...] = ("anthropic", "openai")
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
    # The reasoning efforts it takes, least first, and its default: what
    # it is asked for until the person moves it. Empty where it takes none.
    efforts: tuple[str, ...] = ()
    default_effort: str = ""

    def effort_at(self, set_to: str) -> str:
        """The effort to ask it for when the provider's is `set_to`: that,
        where it takes it, else its default."""
        return set_to if set_to in self.efforts else self.default_effort


EFFORTS = ("low", "medium", "high", "xhigh", "max")
NO_MAX = EFFORTS[:-1]
OR_NONE = ("none", *EFFORTS)
OR_NONE_NO_MAX = ("none", *NO_MAX)


# The recommended model of each keyed provider is its default and heads
# its group. Anthropic comes first: Haiku 5.5 is what the CLI starts on.
CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(
        "anthropic",
        "claude-haiku-5-5",
        "Haiku 5.5",
        "the quickest and the cheapest",
        True,
        efforts=EFFORTS,
        # Its API's own is medium. Anthropic advises high for longer agent
        # tasks: at low and medium, in a long agent prompt, it more often
        # stops early or reports a change done without checking it.
        default_effort="high",
    ),
    ModelInfo(
        "anthropic",
        "claude-sonnet-5-5",
        "Sonnet 5.5",
        "speed and intelligence in balance",
        efforts=EFFORTS,
        default_effort="high",
    ),
    ModelInfo(
        "anthropic",
        "claude-opus-5-5",
        "Opus 5.5",
        "long-running agentic coding and knowledge work",
        efforts=EFFORTS,
        default_effort="medium",  # its API's own, one below Opus 5's
    ),
    ModelInfo(
        "anthropic",
        "claude-fable-5-1",
        "Fable 5.1",
        "the most capable: long, hard tasks",
        efforts=EFFORTS,
        default_effort="high",
    ),
    ModelInfo(
        "anthropic",
        "claude-opus-5",
        "Opus 5",
        "the previous Opus",
        efforts=EFFORTS,
        default_effort="high",
    ),
    ModelInfo(
        "anthropic",
        "claude-sonnet-5",
        "Sonnet 5",
        "the previous Sonnet",
        efforts=EFFORTS,
        default_effort="high",
    ),
    ModelInfo("anthropic", "claude-haiku-4-5", "Haiku 4.5", "the previous Haiku"),
    ModelInfo(
        "openai",
        "gpt-6-luna",
        "GPT-6 Luna",
        "efficient: focused, high-volume work",
        True,
        efforts=OR_NONE,
        # Its API's own is medium. Measured on this CLI's agent (2026-10-05),
        # max got hard problems right 20 times in 24 where medium, high and
        # xhigh got 12 or 13, for 1.4x the cost on ordinary tasks and 2.4x on
        # hard ones — of the cheapest model there is.
        default_effort="max",
    ),
    ModelInfo(
        "openai",
        "gpt-5.6-terra",
        "GPT-5.6 Terra",
        "intelligence and cost in balance",
        efforts=OR_NONE,
        default_effort="medium",
    ),
    ModelInfo(
        "openai",
        "gpt-5.6-sol",
        "GPT-5.6 Sol",
        "complex professional work",
        efforts=OR_NONE,
        default_effort="medium",
    ),
    ModelInfo(
        "openai",
        "gpt-6-astra",
        "GPT-6 Astra",
        "the most capable: the hardest work",
        efforts=EFFORTS,
        default_effort="medium",
    ),
    ModelInfo(
        "openai",
        "gpt-5.6-luna",
        "GPT-5.6 Luna",
        "the previous Luna",
        efforts=OR_NONE,
        default_effort="medium",
    ),
    ModelInfo(
        "openai",
        "gpt-5.5",
        "GPT-5.5",
        "coding and professional work",
        efforts=OR_NONE_NO_MAX,
        default_effort="medium",
    ),
    ModelInfo(
        "openai",
        "gpt-5.4",
        "GPT-5.4",
        "the previous generation",
        efforts=OR_NONE_NO_MAX,
        default_effort="low",  # the API's own is `none`, as for the two below
    ),
    ModelInfo(
        "openai",
        "gpt-5.4-mini",
        "GPT-5.4 mini",
        "small and quick",
        efforts=OR_NONE_NO_MAX,
        default_effort="low",
    ),
    ModelInfo(
        "openai",
        "gpt-5.4-nano",
        "GPT-5.4 nano",
        "the cheapest: simple, high-volume tasks",
        efforts=OR_NONE_NO_MAX,
        default_effort="low",
    ),
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
