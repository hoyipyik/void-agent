"""The model a suite names, made with the key the environment holds — the
one place in the evals a provider's SDK is imported, lazily.

What a request asks for is exactly what the suite says: its effort, and
its `extra` merged in beside it. Nothing is checked against a catalogue
or replaced by a default the suite did not name; a level the model does
not take is the provider's to refuse, and the run says so."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from void_agent import Llm
from void_agent.evals.suite import ModelSpec, SuiteError

Close = Callable[[], Awaitable[None]]
# Room for the thinking a high effort spends (the CLI's own ceiling).
THINKING_MAX_TOKENS = 64_000
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _merged(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merged(merged[key], cast("Mapping[str, Any]", value))
        else:
            merged[key] = value
    return merged


def request_extra(spec: ModelSpec) -> dict[str, Any]:
    """What every request adds: the effort, the way this provider takes
    one, and the suite's `extra` over it."""
    base: dict[str, Any] = {}
    if spec.effort and spec.provider == "openai":
        base = {"reasoning": {"effort": spec.effort}}
    elif spec.effort and spec.provider == "anthropic":
        base = {"thinking": {"type": "adaptive"}, "output_config": {"effort": spec.effort}}
    return _merged(base, spec.extra)


def make_llm(spec: ModelSpec, env: Mapping[str, str]) -> tuple[Llm, Close]:
    """The model, and how to close its client when the run is over."""
    key_name = KEYS.get(spec.provider)
    key = env.get(key_name, "") if key_name else ""
    if key_name and not key:
        raise SuiteError(f"the model is {spec.provider}'s, and {key_name} is not set")
    extra = request_extra(spec) or None
    if spec.provider == "openai":
        from openai import AsyncOpenAI

        from void_agent.providers.openai_responses import OpenAiResponsesLlm

        client = AsyncOpenAI(api_key=key, base_url=env.get("OPENAI_BASE_URL") or None)
        return OpenAiResponsesLlm(spec.model, client=client, extra=extra), client.close
    if spec.provider == "anthropic":
        from anthropic import AsyncAnthropic

        from void_agent.providers.anthropic import AnthropicLlm

        anthropic = AsyncAnthropic(api_key=key)
        if not spec.effort:
            return AnthropicLlm(spec.model, client=anthropic, extra=extra), anthropic.close
        llm = AnthropicLlm(
            spec.model, client=anthropic, max_tokens=THINKING_MAX_TOKENS, extra=extra
        )
        return llm, anthropic.close
    if spec.provider == "ollama":
        from openai import AsyncOpenAI

        from void_agent.providers.openai import OpenAiLlm

        host = env.get("OLLAMA_HOST") or "http://localhost:11434"
        local = AsyncOpenAI(api_key="ollama", base_url=f"{host.rstrip('/')}/v1")
        return OpenAiLlm(spec.model, client=local), local.close
    raise SuiteError(f"the model's provider is openai, anthropic or ollama, not {spec.provider!r}")
