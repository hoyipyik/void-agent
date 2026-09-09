"""Which agent the CLI runs: one of the scanned ones.

The CLI is a chat client of the runtime, not an agent. The agents it can
run are read off folders, one level deep and in this order: the built-in
shelf (`cli/agents/`), `~/.void/agents`, and every `--workspace` folder.
A module with a `build_agent(llm) -> Agent` is an agent: its file's stem
is the name in `/agent`, the first line of its docstring the blurb. A
module without one is a helper — its siblings import it as
`from . import helper` — and is never listed. A file that cannot be
imported is listed with the reason, and the rest mount. A name taken by
an earlier source is suffixed `-1`, `-2` in source order, so nothing
replaces anything: the picker shows both, each with its source and its
blurb.

Every loaded agent can reach every other by name. A builder that takes a
second argument is handed the pool: `agents("writer")` is the writer
built on the same model — the one beside the caller if there is one,
else the pool's by id (`writer-1` names the one the picker calls that).
`agents.mounted` is what the process mounted, the MCP tools and the
skills, for a builder that wants them on a sub-agent; the root gets them
from the registry regardless. A cycle is an error with its path, a name
that is not there an error naming what is, and both surface at scan,
where every builder is run once on a scripted model — so `/agent` shows
a broken graph before anyone picks from it.

The agent is rebuilt every turn from the config, so a switch takes
effect at once. Whatever a builder's own entry point expects, the CLI
hands it the session as history — the CLI is a chat. Without a provider
a scripted model says so and the shell still runs.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from cli.agents.universal import chat
from cli.config import DEFAULT_AGENT, Config
from cli.llm import resolve_llm
from cli.mcp.bench import Bench
from void_agent import Agent, Llm, ScriptedLlm, Tool, say
from void_agent.skills import SkillInfo, tools_for

AGENTS_DIR = "agents"
ENTRY_POINT = "build_agent"
# Where a folder's modules are imported: `void_agents.<source key>.<stem>`,
# so two folders may both hold a writer.py and neither shadows the other.
NAMESPACE = "void_agents"

Builder = Callable[..., Agent]

NO_PROVIDER = (
    "No provider configured. `/model` picks one — a cloud model asks for its key, an"
    " Ollama model needs none — and `/key` adds a key; or set `ANTHROPIC_API_KEY`,"
    " `OPENAI_API_KEY` or `OLLAMA_MODEL` in the environment and start again."
)


class AgentLoadError(Exception):
    """A source that cannot be read, or a name that cannot be built."""


@dataclass(frozen=True, slots=True)
class Source:
    """One place agents are read from: a package (the built-in shelf) or a
    folder on disk. `key` is its module namespace, `label` what the
    picker shows."""

    key: str
    label: str
    package: str | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        if (self.package is None) == (self.path is None):
            raise ValueError("a source is a package or a path, one of the two")


BUILTIN = Source("builtin", "built-in", package="cli.agents")


def folder(path: Path, *, label: str, key: str | None = None) -> Source:
    """A folder of agents. The key defaults to the label, made a name."""
    return Source(key or re.sub(r"\W+", "_", label).strip("_") or "folder", label, path=path)


@dataclass(frozen=True, slots=True)
class AgentInfo:
    """One agent as the person sees it: its name in `/agent`, where it
    came from, its blurb, and why it cannot run — if it cannot."""

    id: str
    source: str
    blurb: str
    error: str | None = None


@dataclass(slots=True)
class _Entry:
    info: AgentInfo
    stem: str
    source: Source
    builder: Builder | None
    takes_pool: bool


class Pool:
    """What a builder is handed beside the model, when it takes a second
    argument: the other agents by name, built on the same model, and the
    tools the process mounted."""

    def __init__(
        self,
        registry: Registry,
        llm: Llm,
        mounted: tuple[Tool, ...],
        *,
        source: Source,
        stack: tuple[str, ...],
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._mounted = mounted
        self._source = source
        self._stack = stack

    def __call__(self, name: str) -> Agent:
        """The named agent, built: the one beside the caller if there is
        one, else the pool's by id. A cycle is an error with its path."""
        entry = self._registry.resolve(name, self._source)
        if entry is None:
            there = ", ".join(info.id for info in self._registry.entries)
            raise AgentLoadError(f"no agent named `{name}` — there are: {there}")
        if entry.info.error is not None:
            raise AgentLoadError(f"`{name}` cannot load: {entry.info.error}")
        if entry.info.id in self._stack:
            raise AgentLoadError(f"a cycle: {' → '.join((*self._stack, entry.info.id))}")
        try:
            return self._registry.build(
                entry, self._llm, self._mounted, stack=(*self._stack, entry.info.id)
            )
        except AgentLoadError as error:
            raise AgentLoadError(f"`{name}` cannot load: {error}") from error
        except Exception as error:
            raise AgentLoadError(
                f"`{name}` cannot load: {type(error).__name__}: {error}"
            ) from error

    @property
    def mounted(self) -> tuple[Tool, ...]:
        """The MCP tools and the skills the process mounted, as the
        session marked them."""
        return self._mounted


class Registry:
    """The agents that exist for this process: what the sources hold, and
    the MCP servers and skills every one of them draws on."""

    def __init__(
        self,
        sources: Iterable[Source] = (BUILTIN,),
        bench: Bench | None = None,
        skills: Sequence[SkillInfo] = (),
    ) -> None:
        self._sources = tuple(sources)
        keys = [source.key for source in self._sources]
        if len(set(keys)) != len(keys):
            raise ValueError(f"two sources share a key: {keys}")
        self._entries: dict[str, _Entry] = {}
        self.bench = bench
        # The shelf as it was last read; the app refreshes it, so nothing
        # here touches the disk on a turn.
        self.skills: Sequence[SkillInfo] = skills
        self.scan()

    # ── what there is ─────────────────────────────────────────────────────

    @property
    def sources(self) -> tuple[Source, ...]:
        return self._sources

    @property
    def entries(self) -> tuple[AgentInfo, ...]:
        return tuple(entry.info for entry in self._entries.values())

    def describe(self, name: str) -> AgentInfo | None:
        entry = self._entries.get(name)
        return entry.info if entry is not None else None

    def label(self, name: str) -> str:
        return name

    def summary(self) -> str:
        """The agents in one line, for `/status`."""
        count = len(self._entries)
        broken = sum(1 for entry in self._entries.values() if entry.info.error is not None)
        line = f"{count} mounted"
        return f"{line}, {broken} cannot load" if broken else line

    def skills_summary(self, config: Config) -> str:
        """The shelf in one line, for `/status`."""
        shelf = tuple(self.skills)
        if not shelf:
            return "none"
        on = len(config.enabled_skills(info.id for info in shelf))
        return f"{on} of {len(shelf)} on"

    def startup(self, requested: str | None, saved: str) -> str:
        """The agent to start on: what `--agent` / `VOID_AGENT` asks for —
        an error if it is not here — else the saved choice while it is
        still here, else the default."""
        if requested:
            if requested not in self._entries:
                there = ", ".join(self._entries)
                raise AgentLoadError(f"no agent named `{requested}` — there are: {there}")
            return requested
        return saved if saved in self._entries else DEFAULT_AGENT

    # ── the scan ──────────────────────────────────────────────────────────

    def scan(self) -> None:
        """Read every source again. A module already imported is kept as
        it is — an edit needs a restart — a new file appears, a file that
        is gone goes, and one that failed to import is tried again. Then
        every builder is run once on a scripted model, so a missing peer,
        a cycle or a crash is on the entry before anyone picks it."""
        entries: dict[str, _Entry] = {}
        counts: dict[str, int] = {}
        for source in self._sources:
            for stem, loaded in self._read(source):
                if isinstance(loaded, types.ModuleType) and not hasattr(loaded, ENTRY_POINT):
                    continue  # a helper: its siblings import it, the picker never lists it
                taken = counts.get(stem, 0)
                counts[stem] = taken + 1
                agent_id = stem if taken == 0 else f"{stem}-{taken}"
                entries[agent_id] = _entry(agent_id, stem, source, loaded)
        self._entries = entries
        for entry in entries.values():
            if entry.builder is None or entry.info.error is not None:
                continue
            try:
                self.build(entry, ScriptedLlm([]), (), stack=(entry.info.id,))
            except AgentLoadError as error:
                entry.info = replace(entry.info, error=str(error))
            except Exception as error:  # the builder is the person's code
                entry.info = replace(entry.info, error=f"{type(error).__name__}: {error}")

    def _read(self, source: Source) -> list[tuple[str, types.ModuleType | str]]:
        """Every module the source holds, the default agent's stem first
        and the rest by name: imported, or the reason it could not be."""
        if source.package is not None:
            root = importlib.import_module(source.package)
            names = [
                name
                for _, name, _ in pkgutil.iter_modules(root.__path__)
                if not name.startswith("_")
            ]
            candidates = [(name, f"{source.package}.{name}") for name in _ordered(names)]
            return [(name, _import_module(qualified)) for name, qualified in candidates]
        path = source.path
        assert path is not None
        if not path.exists():
            return []
        if not path.is_dir():
            raise AgentLoadError(f"{source.label} is not a folder")
        files: dict[str, Path] = {}
        for child in sorted(path.iterdir()):
            if child.name.startswith(("_", ".")):
                continue
            if child.is_file() and child.suffix == ".py" and child.stem.isidentifier():
                files[child.stem] = child
            elif child.is_dir() and child.name.isidentifier():
                init = child / "__init__.py"
                if init.is_file():
                    files[child.name] = init
        return [(stem, _import_file(source, stem, files[stem])) for stem in _ordered(files)]

    # ── the build ─────────────────────────────────────────────────────────

    def resolve(self, name: str, source: Source) -> _Entry | None:
        """The entry a name means from inside `source`: its own by stem
        first, then the pool's by id."""
        for entry in self._entries.values():
            if entry.source.key == source.key and entry.stem == name:
                return entry
        return self._entries.get(name)

    def build(
        self, entry: _Entry, llm: Llm, mounted: tuple[Tool, ...], *, stack: tuple[str, ...]
    ) -> Agent:
        """One entry's builder, run: on the model, with the pool beside it
        if it takes one. `stack` is the chain of builds this one is in,
        the pool's guard against a cycle."""
        assert entry.builder is not None
        pool = Pool(self, llm, mounted, source=entry.source, stack=stack)
        agent = entry.builder(llm, pool) if entry.takes_pool else entry.builder(llm)
        if not isinstance(agent, Agent):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise AgentLoadError(
                f"{ENTRY_POINT} returned {type(agent).__name__}, not an Agent"  # pyright: ignore[reportUnreachable]
            )
        return agent

    def build_agent(self, config: Config) -> Agent:
        """One turn's agent: the chosen one on the configured provider,
        the mounted tools added. What cannot be built is a scripted
        model saying why."""
        entry = self._entries.get(config.agent)
        if entry is None:
            return chat(
                ScriptedLlm(
                    [say(f"No agent named `{config.agent}` is mounted. `/agent` picks one.")]
                )
            )
        if entry.info.error is not None:
            return _cannot_load(config.agent, entry.info.error)
        llm = resolve_llm(config) or ScriptedLlm([say(NO_PROVIDER)])
        mounted = self._mounted(config)
        try:
            agent = self.build(entry, llm, mounted, stack=(entry.info.id,))
        except AgentLoadError as error:
            return _cannot_load(config.agent, str(error))
        except Exception as error:  # the builder is the person's code
            return _cannot_load(config.agent, f"{type(error).__name__}: {error}")
        agent.prompt(lambda history: list(history))
        return self._with_mounted(agent, mounted)

    def _with_mounted(self, agent: Agent, mounted: tuple[Tool, ...]) -> Agent:
        """What was mounted for the process, on whichever agent is running:
        the MCP tools and the skills. Mounting is the process's business and
        the same for every agent — what `/mcp` and `/skill` marked is the
        session's."""
        registered = set(agent.tool_names)
        for capability in mounted:
            # An agent that already has the name keeps its own tool — its
            # builder took it from the pool, or chose a name that collides.
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


def _cannot_load(name: str, reason: str) -> Agent:
    return chat(ScriptedLlm([say(f"The agent `{name}` cannot load: {reason}.")]))


def _ordered(stems: Iterable[str]) -> list[str]:
    """The default agent first, then by name."""
    return sorted(stems, key=lambda stem: (stem != DEFAULT_AGENT, stem))


def _entry(agent_id: str, stem: str, source: Source, loaded: types.ModuleType | str) -> _Entry:
    """An entry from a module, or from the reason it could not be imported."""
    if isinstance(loaded, str):
        return _Entry(AgentInfo(agent_id, source.label, "", loaded), stem, source, None, False)
    builder = getattr(loaded, ENTRY_POINT)
    blurb = _first_line(getattr(builder, "__doc__", None)) or _first_line(loaded.__doc__)
    if not callable(builder):
        info = AgentInfo(agent_id, source.label, blurb, f"{ENTRY_POINT} is not callable")
        return _Entry(info, stem, source, None, False)
    info = AgentInfo(agent_id, source.label, blurb)
    return _Entry(info, stem, source, cast("Builder", builder), _takes_pool(builder))


def _first_line(doc: str | None) -> str:
    lines = (doc or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _takes_pool(builder: Callable[..., Any]) -> bool:
    """Whether the builder has a second positional place, for the pool."""
    try:
        parameters = list(inspect.signature(builder).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return True
    return sum(1 for p in parameters if p.kind in positional) >= 2


def _import_module(qualified: str) -> types.ModuleType | str:
    """A package's module, imported — or the reason it could not be."""
    try:
        return importlib.import_module(qualified)
    except Exception as error:  # whatever the module did at import
        return f"{type(error).__name__}: {error}"


def _import_file(source: Source, stem: str, file: Path) -> types.ModuleType | str:
    """A folder's module, imported under the source's namespace so its
    siblings are `from . import x` — or the reason it could not be. One
    already imported from the same file is kept as it is."""
    assert source.path is not None
    qualified = f"{NAMESPACE}.{source.key}.{stem}"
    cached = sys.modules.get(qualified)
    if cached is not None and getattr(cached, "__file__", None) == str(file):
        return cached
    _package(NAMESPACE, [])
    _package(f"{NAMESPACE}.{source.key}", [str(source.path)])
    is_package = file.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        qualified, file, submodule_search_locations=[str(file.parent)] if is_package else None
    )
    if spec is None or spec.loader is None:
        return f"cannot import {file.name}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # whatever the module did at import
        sys.modules.pop(qualified, None)
        return f"{type(error).__name__}: {error}"
    return module


def _package(name: str, path: list[str]) -> None:
    """A package that exists only to hold a source's modules."""
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__package__ = name
        sys.modules[name] = package
    package.__path__ = path


_default: Registry | None = None


def default_registry() -> Registry:
    """The built-in shelf alone, scanned once, on first use: what an app
    given no registry runs on. Nothing scans at import — the toolbox
    subprocess imports this module too, and should start without reading
    a single agent."""
    global _default
    if _default is None:
        _default = Registry()
    return _default
