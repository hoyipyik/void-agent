"""A real MCP server over HTTP, in this process.

What a hosted server looks like to the bridge: stateless, answering in
JSON, behind a bearer token. The in-memory `FakeMcp` skips the transport,
and the transport is where a refused token, a wrong address and a server
that never answers live — so these are served on a socket of their own.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server.lowlevel import Server
from mcp.types import CallToolResult, ListToolsResult, TextContent
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp import Tool as McpTool

# The examples a publisher's own script is tested on, byte for byte.
VECTORS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "mcp_fingerprint_vectors.json").read_text(encoding="utf-8")
)
TOKEN = "wb_pat_the-one-that-works"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


class Published:
    """A server that publishes the shared examples as its tools, behind a
    bearer token, and remembers every token it was shown. A call to one of
    `stalls` is taken and not answered until `release()`."""

    def __init__(self, stalls: frozenset[str] = frozenset()) -> None:
        self.shown: list[str | None] = []
        self._released = asyncio.Event()
        tools = [McpTool.model_validate(example["tool"]) for example in VECTORS["tools"]]

        async def list_tools(context: Any, params: Any) -> ListToolsResult:
            return ListToolsResult(tools=tools)

        async def call_tool(context: Any, params: Any) -> CallToolResult:
            if params.name in stalls:
                await self._released.wait()
            return CallToolResult(content=[TextContent(type="text", text="pong")])

        server = Server(
            "published",
            instructions=VECTORS["instructions"]["text"],
            on_list_tools=list_tools,
            on_call_tool=call_tool,
        )
        self._mcp: ASGIApp = server.streamable_http_app(json_response=True, stateless_http=True)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            shown = dict(scope["headers"]).get(b"authorization")
            self.shown.append(None if shown is None else shown.decode())
            if shown != BEARER["Authorization"].encode():
                await _answer(send, 401)
                return
        await self._mcp(scope, receive, send)

    def release(self) -> None:
        """Let every stalled call answer, so the server can close."""
        self._released.set()


def answering(status: int) -> ASGIApp:
    """A server that answers everything with one status and no MCP."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            await _answer(send, status)

    return app


async def silent(scope: Scope, receive: Receive, send: Send) -> None:
    """A server that takes the request and never answers it."""
    if scope["type"] != "http":
        return
    while (await receive())["type"] != "http.disconnect":
        pass


async def _answer(send: Send, status: int) -> None:
    await send({"type": "http.response.start", "status": status, "headers": []})
    await send({"type": "http.response.body", "body": b""})


@asynccontextmanager
async def serving(app: ASGIApp) -> AsyncGenerator[str]:
    """`app` on a port of its own, for the length of the block: its URL.
    The socket listens before the block starts, so a request sent at once
    waits for the server rather than finding nobody there."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", timeout_graceful_shutdown=1))
    serve = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await serve


def nobody_listening() -> str:
    """The URL of a port that was free a moment ago and has no server."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"
