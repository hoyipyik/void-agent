"""The Llm the configured provider gives, and the one place in the CLI a
provider's SDK is imported. The imports stay lazy so the unused SDK is
never touched. Ollama is the OpenAI provider pointed at the local
server's `/v1`: no key, the model one of what it has installed."""

from __future__ import annotations

from cli.config import Config
from void_agent import Llm


def resolve_llm(config: Config) -> Llm | None:
    """The configured provider's model, or None when there is none."""
    if not config.configured():
        return None
    if config.provider == "anthropic":
        from anthropic import AsyncAnthropic

        from void_agent.providers.anthropic import AnthropicLlm

        return AnthropicLlm(
            config.anthropic_model, client=AsyncAnthropic(api_key=config.anthropic_api_key)
        )
    from openai import AsyncOpenAI

    from void_agent.providers.openai import OpenAiLlm

    if config.provider == "ollama":
        # Chat Completions at /v1; the key is ignored there, but the SDK insists on one.
        client = AsyncOpenAI(api_key="ollama", base_url=f"{config.ollama_host}/v1")
        return OpenAiLlm(config.ollama_model, client=client)
    client = AsyncOpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
    extra = (
        {"reasoning_effort": config.openai_reasoning_effort}
        if config.openai_reasoning_effort
        else None
    )
    return OpenAiLlm(config.openai_model, client=client, extra=extra)
