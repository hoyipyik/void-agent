"""An MCP client's shape, in memory: what the bridge and the CLI mount."""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent

from mcp import Tool as McpTool

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
    "required": ["path"],
}


def descriptor(name: str, description: str | None = "writes a file") -> McpTool:
    return McpTool(name=name, description=description, input_schema=WRITE_SCHEMA)


def text_result(text: str, *, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


class FakeMcp:
    """An MCP client's shape: the async context manager, list_tools, call_tool."""

    def __init__(
        self,
        tools: list[McpTool],
        result: CallToolResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.open = False
        self.closed = False
        self.listed = 0
        self._tools = tools
        self._result = result or text_result("written")
        self._failure = failure

    async def __aenter__(self) -> FakeMcp:
        self.open = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def list_tools(self) -> Any:
        self.listed += 1
        return type("ListToolsResult", (), {"tools": self._tools})()

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, arguments))
        if self._failure is not None:
            raise self._failure
        return self._result
