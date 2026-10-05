"""The Llm the configured provider gives, and the one place in the CLI a
provider's SDK is imported. The imports stay lazy so the unused SDK is
never touched. OpenAI speaks the Responses API, where its models take
tools at any reasoning effort. Ollama is the Chat Completions adapter
pointed at the local server's `/v1`: no key, the model one of what it has
installed."""

from __future__ import annotations

from typing import Any

from cli.config import Config
from void_agent import Llm

# Room for the thinking a high effort spends: the adapter streams, so a
# large ceiling costs nothing until it is used.
THINKING_MAX_TOKENS = 64_000


def extra_for(config: Config) -> dict[str, Any] | None:
    """What the request adds for the configured model: the effort it is
    asked for, where it takes one. A Claude model that takes one thinks
    adaptively — some do only when told — so the effort is how hard it
    thinks, as OpenAI's is."""
    effort = config.effort
    if not effort:
        return None
    if config.provider == "openai":
        return {"reasoning": {"effort": effort}}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def max_tokens_for(config: Config) -> int | None:
    """The ceiling on a Claude answer: raised where the model thinks, None
    — the adapter's own — everywhere else, since a model the catalogue
    lacks may take less than the raised one."""
    return THINKING_MAX_TOKENS if config.provider == "anthropic" and extra_for(config) else None


def resolve_llm(config: Config) -> Llm | None:
    """The configured provider's model, or None when there is none."""
    if not config.configured():
        return None
    if config.provider == "anthropic":
        from anthropic import AsyncAnthropic

        from void_agent.providers.anthropic import AnthropicLlm

        client = AsyncAnthropic(api_key=config.anthropic_api_key)
        max_tokens = max_tokens_for(config)
        if max_tokens is None:
            return AnthropicLlm(config.anthropic_model, client=client)
        return AnthropicLlm(
            config.anthropic_model, client=client, max_tokens=max_tokens, extra=extra_for(config)
        )
    from openai import AsyncOpenAI

    from void_agent.providers.openai import OpenAiLlm

    if config.provider == "ollama":
        # Chat Completions at /v1; the key is ignored there, but the SDK insists on one.
        client = AsyncOpenAI(api_key="ollama", base_url=f"{config.ollama_host}/v1")
        return OpenAiLlm(config.ollama_model, client=client)
    from void_agent.providers.openai_responses import OpenAiResponsesLlm

    client = AsyncOpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
    return OpenAiResponsesLlm(config.openai_model, client=client, extra=extra_for(config))
