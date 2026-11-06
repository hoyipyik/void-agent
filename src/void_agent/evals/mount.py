"""A persona's servers, mounted as that persona, for one case.

Each server is opened with the persona's own credentials, and what it
published at this mount is kept beside the record: its instructions, and
for every tool its description, schema, read-only hint and fingerprint —
so a run says which version of each tool answered it. Of the tools a
server lists, the persona's filters decide which the agent is offered;
the rest are named as withheld. A tool keeps the name its server gives
it, so two servers that both offer one are refused.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

from void_agent import Tool
from void_agent.evals.suite import Persona, ServerSpec, SuiteError, expand, matches
from void_agent.mcp import McpServer

Opener = Callable[[ServerSpec], McpServer]
SIGNED = "the suite marks this tool as signed"


def open_server(spec: ServerSpec) -> McpServer:
    """The real thing: an HTTP session, or a subprocess."""
    if spec.url:
        return McpServer.http(spec.url, headers=spec.headers or None, timeout=spec.timeout)
    return McpServer.stdio(spec.command, *spec.args, env=spec.env or None, timeout=spec.timeout)


def expanded(spec: ServerSpec, env: Mapping[str, str], persona: str) -> ServerSpec:
    """`spec` with every `${NAME}` read from the environment."""

    def each(values: Mapping[str, str]) -> dict[str, str]:
        return {key: expand(value, env, persona) for key, value in values.items()}

    return replace(
        spec,
        url=expand(spec.url, env, persona),
        headers=each(spec.headers),
        command=expand(spec.command, env, persona),
        args=tuple(expand(arg, env, persona) for arg in spec.args),
        env=each(spec.env),
    )


@dataclass(frozen=True, slots=True)
class Mount:
    # (the server it came from, the tool) for each tool the agent is offered.
    tools: tuple[tuple[str, Tool], ...]
    # Per server: its instructions and each tool as it was published.
    catalog: dict[str, Any]
    offered: tuple[str, ...]
    withheld: tuple[str, ...]


class MountError(Exception):
    """A server could not be mounted: which one, and why, in words."""


def _signed(input: Any) -> str:
    return SIGNED


def _published(server: McpServer) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    for name in server.names:
        descriptor = server.describe(name)
        tools.append(
            {
                "name": name,
                "title": descriptor.title,
                "description": descriptor.description,
                "input_schema": descriptor.input_schema,
                "read_only": server.read_only_hint(name),
                "fingerprint": server.fingerprint(name),
            }
        )
    return {
        "instructions": server.instructions,
        "instructions_fingerprint": server.instructions_fingerprint,
        "tools": tools,
    }


def _offered(name: str, persona: Persona) -> bool:
    if persona.only and not matches(name, persona.only):
        return False
    return not matches(name, persona.exclude)


@asynccontextmanager
async def mounted(
    persona: Persona, env: Mapping[str, str], opener: Opener = open_server
) -> AsyncGenerator[Mount]:
    """Every server of `persona`, open until the block ends. Raises
    `MountError` naming the server that could not be mounted, or
    `SuiteError` for a credential that is not set."""
    async with AsyncExitStack() as stack:
        tools: list[tuple[str, Tool]] = []
        catalog: dict[str, Any] = {}
        withheld: list[str] = []
        for name, spec in persona.servers.items():
            spec = expanded(spec, env, persona.name)
            try:
                server = await stack.enter_async_context(opener(spec))
            except Exception as error:
                raise MountError(f"server `{name}` could not be mounted: {error}") from error
            catalog[name] = _published(server)
            for tool in server.names:
                if not _offered(tool, persona):
                    withheld.append(tool)
                    continue
                approval = _signed if matches(tool, spec.signed) else None
                tools.append((name, server.tool(tool, approval=approval)))
        names = [tool.name for _, tool in tools]
        for name in names:
            if names.count(name) > 1:
                raise SuiteError(f"two servers of persona `{persona.name}` offer `{name}`")
        yield Mount(
            tools=tuple(tools),
            catalog=catalog,
            offered=tuple(names),
            withheld=tuple(withheld),
        )
