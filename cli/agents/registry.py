"""Which agent the CLI runs — one of the mounted ones.

The CLI is a chat client of the runtime, not an agent. The agents it can
run are a registry: `universal` (`universal.py` — the model, a plan,
reflection, a question, and whatever MCP servers and skills the process
mounted), `weather` and `dummy-weather` (`weather.py`, the framework's
claim in one agent and the example to read), plus any `module:function`
mounted at start with `--agent` or `VOID_AGENT`, imported right then, so
a module that cannot load fails at the door, not in a turn. `/agent`
chooses among what is mounted and nothing else: a name that is not in
the list is an error at once, never saved.

The agent is rebuilt every turn from the config, so a switch takes
effect at once. Whatever a builder's own entry point expects, the CLI
hands it the session as history — the CLI is a chat. Without a provider
a scripted model says so and the shell still runs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from cli.agents.universal import chat
from cli.config import DEFAULT_AGENT, Config
from cli.llm import resolve_llm
from cli.mcp.bench import Bench
from void_agent import Agent, Llm, ScriptedLlm, Tool, say
from void_agent.skills import SkillInfo, tools_for

Builder = Callable[[Llm], Agent]


@dataclass(frozen=True, slots=True)
class AgentInfo:
    id: str
    name: str
    blurb: str
    spec: str


CATALOG: tuple[AgentInfo, ...] = (
    AgentInfo(
        "universal",
        "Universal",
        "the model, a plan, a question — and whatever MCP you mounted",
        "cli.agents.universal",
    ),
    AgentInfo(
        "weather",
        "Weather",
        "Open-Meteo: forecasts, hours, history — ask it anything about the weather",
        "cli.agents.weather",
    ),
    AgentInfo(
        "dummy-weather",
        "Dummy weather",
        "the weather agent replayed on a scripted model and canned data — no key, no network",
        "cli.agents.weather:dummy",
    ),
)
assert DEFAULT_AGENT in {info.id for info in CATALOG}

NO_PROVIDER = (
    "No provider configured. `/model` picks one — a cloud model asks for its key, an"
    " Ollama model needs none — and `/key` adds a key; or set `ANTHROPIC_API_KEY`,"
    " `OPENAI_API_KEY` or `OLLAMA_MODEL` in the environment and start again."
)


class AgentLoadError(Exception):
    """The spec names nothing that can be imported and called."""


def load_builder(spec: str) -> Builder:
    """Import `module:function` — the function defaults to `build_agent`."""
    module_name, _, attribute = spec.partition(":")
    attribute = attribute or "build_agent"
    if not module_name:
        raise AgentLoadError("an agent is module:function")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise AgentLoadError(f"cannot import {module_name}: {error}") from error
    builder = getattr(module, attribute, None)
    if not callable(builder):
        raise AgentLoadError(f"{module_name} has no callable {attribute}")
    return cast("Builder", builder)


class Registry:
    """The agents that exist for this process: the catalogue, what was
    mounted at start, and the MCP servers and skills every one of them
    draws on."""

    def __init__(
        self,
        entries: Iterable[AgentInfo] = CATALOG,
        bench: Bench | None = None,
        skills: Sequence[SkillInfo] = (),
    ) -> None:
        self._entries: dict[str, AgentInfo] = {info.id: info for info in entries}
        self._builders: dict[str, Builder] = {}
        self.bench = bench
        # The shelf as it was last read; the app refreshes it, so nothing
        # here touches the disk on a turn.
        self.skills: Sequence[SkillInfo] = skills

    @property
    def entries(self) -> tuple[AgentInfo, ...]:
        return tuple(self._entries.values())

    def describe(self, name: str) -> AgentInfo | None:
        return self._entries.get(name)

    def label(self, name: str) -> str:
        info = self.describe(name)
        return info.name if info is not None else name

    def skills_summary(self, config: Config) -> str:
        """The shelf in one line, for `/status`."""
        shelf = tuple(self.skills)
        if not shelf:
            return "none"
        on = len(config.enabled_skills(info.id for info in shelf))
        return f"{on} of {len(shelf)} on"

    def mount(self, spec: str) -> AgentInfo:
        """Import `module:function` now and add it to the list under its
        own spec as the name. What cannot load is an error here, at the
        door."""
        builder = load_builder(spec)
        doc = (builder.__doc__ or "").strip().splitlines()
        blurb = doc[0].strip() if doc else "mounted at start"
        info = AgentInfo(spec, spec, blurb, spec)
        self._entries[spec] = info
        self._builders[spec] = builder
        return info

    def startup(self, requested: str | None, saved: str) -> str:
        """The agent to start on: what `--agent` / `VOID_AGENT` asks for —
        a name in the list, or a `module:function` mounted now (an error
        if it cannot be) — else the saved choice while it is still here,
        else the default."""
        if requested:
            if self.describe(requested) is None:
                self.mount(requested)
            return requested
        return saved if self.describe(saved) is not None else DEFAULT_AGENT

    def builder_for(self, name: str) -> Builder | None:
        """The mounted agent's builder, or None for a name not in the list."""
        info = self.describe(name)
        if info is None:
            return None
        if name not in self._builders:
            self._builders[name] = load_builder(info.spec)
        return self._builders[name]

    def build_agent(self, config: Config) -> Agent:
        """One turn's agent: the chosen one on the configured provider."""
        try:
            builder = self.builder_for(config.agent)
        except AgentLoadError as error:
            return chat(ScriptedLlm([say(f"The agent `{config.agent}` cannot load: {error}.")]))
        if builder is None:
            return chat(
                ScriptedLlm(
                    [say(f"No agent named `{config.agent}` is mounted. `/agent` picks one.")]
                )
            )
        llm = resolve_llm(config) or ScriptedLlm([say(NO_PROVIDER)])
        agent = builder(llm).prompt(lambda history: list(history))
        return self._with_mounted(agent, config)

    def _with_mounted(self, agent: Agent, config: Config) -> Agent:
        """What was mounted for the process, on whichever agent is running:
        the MCP tools and the skills. Mounting is the process's business and
        the same for every agent — what `/mcp` and `/skill` marked is the
        session's."""
        registered = set(agent.tool_names)
        for capability in self._mounted(config):
            # An agent that already has the name keeps its own tool; a
            # qualified name collides only if a builder chose one.
            if capability.name not in registered:
                agent.tool(capability)
                registered.add(capability.name)
        return agent

    def _mounted(self, config: Config) -> tuple[Tool, ...]:
        tools: tuple[Tool, ...] = ()
        if self.bench is not None:
            tools += self.bench.tools(state=lambda info: config.tool_state(info.id, info.default))
        kept = config.enabled_skills(info.id for info in self.skills)
        return tools + tools_for([info for info in self.skills if info.id in kept])


REGISTRY = Registry()
build_agent = REGISTRY.build_agent
