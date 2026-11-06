"""A real MCP server over stdio, for the tests to run as a subprocess:
`ping` answers at once, `numbers` is taken and never answered. A server
run as a command is where a stuck call lives just as it does over HTTP,
so the bridge's timeout is tested on one."""

from __future__ import annotations

from typing import Any

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent

from mcp import Tool as McpTool

TOOLS = [
    McpTool.model_validate({"name": name, "inputSchema": {"type": "object"}})
    for name in ("ping", "numbers")
]


async def list_tools(context: Any, params: Any) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)


async def call_tool(context: Any, params: Any) -> CallToolResult:
    if params.name == "numbers":
        await anyio.sleep_forever()
    return CallToolResult(content=[TextContent(type="text", text="pong")])


async def main() -> None:
    server = Server("stalling", on_list_tools=list_tools, on_call_tool=call_tool)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
