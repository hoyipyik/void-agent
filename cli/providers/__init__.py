"""The providers as the CLI knows them: their names, the cloud catalogue
`/model` lists (`catalog.py`), and the local Ollama server's own list,
read live (`ollama.py`). Nothing here reaches core or an SDK — the Llm
itself is built from the config in `cli/llm.py`, which is the one place
a provider's SDK is imported."""

from cli.providers.catalog import (
    CATALOG,
    DEFAULT_MODELS,
    KEY_VARIABLES,
    KEYED,
    PROVIDER_LABELS,
    PROVIDERS,
    ModelInfo,
    Provider,
    as_provider,
    describe,
    models_for,
    provider_of,
)
from cli.providers.ollama import DEFAULT_HOST, Ollama, OllamaDown, alias, host_url, usable

__all__ = [
    "CATALOG",
    "DEFAULT_HOST",
    "DEFAULT_MODELS",
    "KEYED",
    "KEY_VARIABLES",
    "PROVIDERS",
    "PROVIDER_LABELS",
    "ModelInfo",
    "Ollama",
    "OllamaDown",
    "Provider",
    "alias",
    "as_provider",
    "describe",
    "host_url",
    "models_for",
    "provider_of",
    "usable",
]
