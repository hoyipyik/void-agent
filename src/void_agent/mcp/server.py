"""A mounted MCP server: its tools, discovered once, gated by your code.

Mounting is a lifecycle, not a call. The connection is opened once — a
subprocess or an HTTP session — the tool list is read from it, and it stays
open for as long as the agent may run. That is why this is an async context
manager and not a function: an agent rebuilt every turn must not respawn
the server every turn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, NoReturn, TextIO

from mcp import Client, StdioServerParameters, stdio_client
from mcp import Tool as McpTool
from void_agent.core.tool import Approval, Tool
from void_agent.mcp.result import McpUnknownTool, value_of

NOT_MOUNTED = "the MCP server is not mounted: use `async with McpServer...`"


class McpServer:
    """One MCP server's tools, as this runtime's `Tool`s."""

    def __init__(self, client: Any = None) -> None:
        # None only from `stdio(..., errlog=…)`, which builds its transport
        # in `__aenter__` so the log file's lifetime is the server's.
        self._client = client
        self._session: Any | None = None
        self._stack: AsyncExitStack | None = None
        self._descriptors: dict[str, McpTool] = {}
        # Set only by `stdio(..., errlog=…)`: the transport is built inside
        # `__aenter__`, where the log file's lifetime can be the server's.
        self._parameters: StdioServerParameters | None = None
        self._errlog: Path | TextIO | None = None

    @classmethod
    def stdio(
        cls,
        command: str,
        *arguments: str,
        env: Mapping[str, str] | None = None,
        errlog: Path | TextIO | None = None,
    ) -> McpServer:
        """A server run as a subprocess: `McpServer.stdio("npx", "-y", "…")`.

        `errlog` is where the child's own noise goes. Left out, it goes to
        this process's stderr, which is right for a script and wrong for a
        terminal UI — there it is the canvas being drawn on. A `Path` is
        opened for the life of the server and closed with it."""
        parameters = StdioServerParameters(
            command=command, args=list(arguments), env=dict(env) if env is not None else None
        )
        if errlog is None:
            return cls(Client(parameters))
        server = cls()
        server._parameters, server._errlog = parameters, errlog
        return server

    @classmethod
    def http(cls, url: str, *, headers: Mapping[str, str] | None = None) -> McpServer:
        """A server reached over HTTP. `headers` carries whatever it wants
        for authentication — a bearer token, an API key — on every request."""
        if not headers:
            return cls(Client(url))
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        return cls(
            Client(
                streamable_http_client(
                    url, http_client=create_mcp_http_client(headers=dict(headers))
                )
            )
        )

    async def __aenter__(self) -> McpServer:
        stack = AsyncExitStack()
        try:
            client = self._client
            if self._parameters is not None and self._errlog is not None:
                errlog: TextIO
                if isinstance(self._errlog, Path):
                    self._errlog.parent.mkdir(parents=True, exist_ok=True)
                    errlog = stack.enter_context(self._errlog.open("w", encoding="utf-8"))
                else:
                    errlog = self._errlog
                client = Client(stdio_client(self._parameters, errlog=errlog))
            if client is None:
                raise RuntimeError("an McpServer needs a client or stdio parameters")
            self._session = await stack.enter_async_context(client)
            self._descriptors = {tool.name: tool for tool in await self._discover()}
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, *args: object) -> None:
        """Closing never suppresses: a failure inside the block is the
        caller's, and the connection goes down either way."""
        stack, self._stack, self._session = self._stack, None, None
        self._descriptors = {}
        if stack is not None:
            await stack.aclose()

    @property
    def names(self) -> tuple[str, ...]:
        """What the server offers, in the order it listed them."""
        return tuple(self._descriptors)

    def describe(self, name: str) -> McpTool:
        """The server's own descriptor — its title, schema and hints. The
        hints are the server's word: read them, then decide the approval
        yourself."""
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            self._refuse(name)
        return descriptor

    def tool(
        self, name: str, *, approval: Approval | None = None, alias: str | None = None
    ) -> Tool:
        """One of the server's tools, with the gate you declare for it.
        `alias` is the name the model sees — the server is always called by
        the name it knows itself."""
        if self._session is None:
            raise RuntimeError(NOT_MOUNTED)
        descriptor = self.describe(name)
        session = self._session

        async def handler(input: dict[str, Any]) -> Any:
            return value_of(await session.call_tool(name, input), name)

        return Tool(
            name=alias or name,
            description=descriptor.description or descriptor.title or name,
            handler=handler,
            approval=approval,
            input_schema=descriptor.input_schema,
        )

    def tools(
        self, *, approvals: Mapping[str, Approval] | None = None, prefix: str = ""
    ) -> tuple[Tool, ...]:
        """Every tool the server offers. `approvals` names the ones a person
        must sign — a name the server does not have is an error here, at
        mount time, not on the call that would have been ungated. `prefix`
        namespaces them, so two servers that both offer `search` can be
        mounted on one agent."""
        approvals = approvals or {}
        for name in approvals:
            if name not in self._descriptors:
                self._refuse(name)
        return tuple(
            self.tool(name, approval=approvals.get(name), alias=f"{prefix}{name}" or None)
            for name in self._descriptors
        )

    async def _discover(self) -> Sequence[McpTool]:
        """Every page of the tool list."""
        assert self._session is not None
        found: list[McpTool] = []
        cursor: str | None = None
        while True:
            page = (
                await self._session.list_tools()
                if cursor is None
                else await self._session.list_tools(cursor=cursor)
            )
            found.extend(page.tools)
            cursor = getattr(page, "next_cursor", None)
            if cursor is None:
                return found

    def _refuse(self, name: str) -> NoReturn:
        if self._session is None and not self._descriptors:
            raise RuntimeError(NOT_MOUNTED)
        raise McpUnknownTool(
            f"the server has no tool `{name}`; it offers: {', '.join(self.names)}"
        )
