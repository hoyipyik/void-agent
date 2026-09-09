"""Where the provider comes from.

The environment names it first — ANTHROPIC_API_KEY / ANTHROPIC_MODEL,
OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_REASONING_EFFORT,
the variables the reference server reads, and OLLAMA_HOST / OLLAMA_MODEL
for a local Ollama — and the config file (`~/.void/config.json`, written
by the key prompt and `/model`, private to the user) fills in what the
environment lacks. The provider object itself is built from a `Config`
in `cli/llm.py`, the one place an SDK is imported.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from cli.providers.catalog import DEFAULT_MODELS, PROVIDERS, Provider, as_provider
from cli.providers.ollama import DEFAULT_HOST, host_url

DEFAULT_AGENT = "universal"

# What the person did in `/mcp` and `/skill`. "on" is the absence of a
# mark, so a server that grows a tool, or a shelf that grows a skill,
# offers it without asking again.
ToolState = Literal["on", "off", "signed"]
Switch = Literal["on", "off"]
_MODEL_FIELDS: dict[Provider, str] = {
    "anthropic": "anthropic_model",
    "openai": "openai_model",
    "ollama": "ollama_model",
}
_KEY_FIELDS: dict[Provider, str] = {  # the keyed providers only
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
}


@dataclass(frozen=True, slots=True)
class Config:
    provider: Provider | None = None
    anthropic_api_key: str = ""
    anthropic_model: str = DEFAULT_MODELS["anthropic"]
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = DEFAULT_MODELS["openai"]
    # OpenAI's reasoning-tier models reject function tools on Chat
    # Completions unless this is "none" — the server reads the same variable.
    openai_reasoning_effort: str = ""
    # Ollama: a local server, no key. Its models are whatever it has
    # installed, so there is no default — choosing one is what configures it.
    ollama_host: str = DEFAULT_HOST
    ollama_model: str = ""
    # The agent the CLI runs, by its name in the registry (`cli/agents/`).
    agent: str = DEFAULT_AGENT
    # What `/mcp` and `/skill` marked: the servers never started, the tools
    # kept from the model, the ones that must be signed, and the skills
    # left off the shelf. Everything not named here is simply on.
    mcp_off_servers: tuple[str, ...] = ()
    mcp_on: tuple[str, ...] = ()
    mcp_off: tuple[str, ...] = ()
    mcp_signed: tuple[str, ...] = ()
    skills_off: tuple[str, ...] = ()

    def server_state(self, name: str) -> Switch:
        return "off" if name in self.mcp_off_servers else "on"

    def with_server_state(self, name: str, state: Switch) -> Config:
        kept = tuple(n for n in self.mcp_off_servers if n != name)
        return replace(
            self, mcp_off_servers=tuple(sorted((*kept, name))) if state == "off" else kept
        )

    def skill_state(self, skill_id: str) -> Switch:
        return "off" if skill_id in self.skills_off else "on"

    def with_skill_state(self, skill_id: str, state: Switch) -> Config:
        kept = tuple(n for n in self.skills_off if n != skill_id)
        return replace(
            self, skills_off=tuple(sorted((*kept, skill_id))) if state == "off" else kept
        )

    def enabled_skills(self, known: Iterable[str]) -> frozenset[str]:
        """The skills to build, by id: everything on the shelf but what was
        turned off."""
        return frozenset(skill_id for skill_id in known if self.skill_state(skill_id) == "on")

    def running_servers(self, known: Iterable[str]) -> tuple[str, ...]:
        """The servers to start, in the file's order."""
        return tuple(name for name in known if self.server_state(name) == "on")

    def tool_state(self, tool_id: str, fallback: ToolState = "signed") -> ToolState:
        """What this tool is, for this session. A mark the person made
        outranks everything; a tool nobody has judged falls to the state
        its server was given (`cli/mcp.py`), which is `signed` unless the
        person's own file says otherwise."""
        if tool_id in self.mcp_signed:
            return "signed"
        if tool_id in self.mcp_off:
            return "off"
        return "on" if tool_id in self.mcp_on else fallback

    def with_tool_state(self, tool_id: str, state: ToolState) -> Config:
        """One tool's state, marked explicitly — including "on", which is
        no longer the absence of a mark: it is the person allowing a tool
        their server would otherwise have had signed."""
        marks = {
            "on": tuple(name for name in self.mcp_on if name != tool_id),
            "off": tuple(name for name in self.mcp_off if name != tool_id),
            "signed": tuple(name for name in self.mcp_signed if name != tool_id),
        }
        marks[state] = tuple(sorted((*marks[state], tool_id)))
        return replace(self, mcp_on=marks["on"], mcp_off=marks["off"], mcp_signed=marks["signed"])

    def key_for(self, provider: Provider) -> str:
        """The key held for a provider; Ollama holds none."""
        if provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key if provider == "openai" else ""

    def model_for(self, provider: Provider) -> str:
        if provider == "anthropic":
            return self.anthropic_model
        return self.openai_model if provider == "openai" else self.ollama_model

    def ready(self, provider: Provider) -> bool:
        """Whether the provider can run: a keyed one has its key, Ollama
        has a model chosen."""
        if provider == "ollama":
            return bool(self.ollama_model)
        return bool(self.key_for(provider))

    @property
    def api_key(self) -> str:
        return self.key_for(self.provider) if self.provider is not None else ""

    @property
    def model(self) -> str:
        return self.model_for(self.provider) if self.provider is not None else ""

    def configured(self) -> bool:
        return self.provider is not None and self.ready(self.provider)

    def with_model(self, provider: Provider, model: str) -> Config:
        return replace(self, provider=provider, **{_MODEL_FIELDS[provider]: model})

    def with_key(self, provider: Provider, key: str) -> Config:
        """A keyed provider's key; Ollama has no field for one."""
        return replace(self, provider=provider, **{_KEY_FIELDS[provider]: key})

    def with_agent(self, agent: str) -> Config:
        return replace(self, agent=agent)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Config:
        defaults = cls()
        return cls(
            provider=as_provider(data.get("provider")),
            anthropic_api_key=str(data.get("anthropic_api_key") or ""),
            anthropic_model=str(data.get("anthropic_model") or defaults.anthropic_model),
            openai_api_key=str(data.get("openai_api_key") or ""),
            openai_base_url=str(data["openai_base_url"]) if data.get("openai_base_url") else None,
            openai_model=str(data.get("openai_model") or defaults.openai_model),
            openai_reasoning_effort=str(data.get("openai_reasoning_effort") or ""),
            ollama_host=host_url(str(data.get("ollama_host") or "")),
            ollama_model=str(data.get("ollama_model") or ""),
            agent=str(data.get("agent") or defaults.agent),
            mcp_off_servers=_names(data.get("mcp_off_servers")),
            mcp_on=_names(data.get("mcp_on")),
            mcp_off=_names(data.get("mcp_off")),
            mcp_signed=_names(data.get("mcp_signed")),
            skills_off=_names(data.get("skills_off")),
        )


def _names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(name) for name in cast("list[Any]", value))


def _read(path: Path) -> Config:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Config()
    return Config.from_json(cast("dict[str, Any]", data)) if isinstance(data, dict) else Config()


def load_config(env: Mapping[str, str], path: Path) -> Config:
    """The file's settings with the environment laid over them, field by
    field. The provider is the file's choice when it is still ready — its
    key held, or for Ollama a model chosen — else the first that is, the
    cloud providers before the local one."""
    base = _read(path)
    config = replace(
        base,
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or base.anthropic_api_key,
        anthropic_model=env.get("ANTHROPIC_MODEL") or base.anthropic_model,
        openai_api_key=env.get("OPENAI_API_KEY") or base.openai_api_key,
        openai_model=env.get("OPENAI_MODEL") or base.openai_model,
        openai_base_url=env.get("OPENAI_BASE_URL") or base.openai_base_url,
        openai_reasoning_effort=env.get("OPENAI_REASONING_EFFORT") or base.openai_reasoning_effort,
        ollama_host=host_url(env.get("OLLAMA_HOST") or base.ollama_host),
        ollama_model=env.get("OLLAMA_MODEL") or base.ollama_model,
    )
    chosen = base.provider if base.provider is not None and config.ready(base.provider) else None
    if chosen is None:
        chosen = next((p for p in PROVIDERS if config.ready(p)), None)
    return replace(config, provider=chosen)


def save_config(path: Path, config: Config) -> None:
    """Written private to the user (0600): it holds a key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(config), indent=2)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.chmod(path, 0o600)
