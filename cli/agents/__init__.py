"""The agents the CLI can run, and which one it does.

`registry.py` is the list and the choice: the catalogue, what `--agent`
mounted at start, and the MCP servers and skills every agent draws on.
`universal.py` is the CLI's own agent; `weather.py` is the example — a
real agent on a real API, the framework's claim in one file, and the one
to read before writing your own. An agent is any `build_agent(llm)`; the
catalogue names them as `module:function` strings, the shape `--agent`
takes.
"""

from cli.agents.registry import (
    CATALOG,
    NO_PROVIDER,
    REGISTRY,
    AgentInfo,
    AgentLoadError,
    Builder,
    Registry,
    build_agent,
    load_builder,
)

__all__ = [
    "CATALOG",
    "NO_PROVIDER",
    "REGISTRY",
    "AgentInfo",
    "AgentLoadError",
    "Builder",
    "Registry",
    "build_agent",
    "load_builder",
]
