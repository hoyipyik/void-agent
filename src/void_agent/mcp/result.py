"""What a server said, as a tool's result.

A call's outcome is classified the way every other tool's is: a server
that reports an error speaks to the model (`Rejected`), and a transport
that breaks does not — it raises, and `Tool.invoke` files it as `Internal`.

Binary blocks are described, never inlined: a tool result becomes
transcript text, so pasting a base64 image into it would spend the
context window on bytes the model cannot read.
"""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, EmbeddedResource, ImageContent, ResourceLink, TextContent

from void_agent.core.errors import Rejected


class McpUnknownTool(LookupError):
    """A tool name the mounted server does not have."""


def value_of(result: CallToolResult, tool: str) -> Any:
    """The call's result, or `Rejected` in the server's own words."""
    if result.is_error:
        raise Rejected(f"{tool}: {_text(result) or 'the MCP server reported an error'}")
    if result.structured_content is not None:
        return _unwrapped(result.structured_content)
    blocks = [_block(block) for block in result.content]
    if not blocks:
        return ""
    if len(blocks) == 1 and isinstance(blocks[0], str):
        return blocks[0]
    return blocks


def _unwrapped(structured: dict[str, Any]) -> Any:
    """A server wraps a return that is not an object in `{"result": …}`.
    The model wants the value, not the envelope."""
    if len(structured) == 1 and "result" in structured:
        return structured["result"]
    return structured


def _text(result: CallToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent)).strip()


def _block(block: object) -> Any:
    if isinstance(block, TextContent):
        return block.text
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "media_type": block.mime_type,
            "bytes": _decoded_length(block.data),
        }
    if isinstance(block, ResourceLink):
        return {"type": "resource_link", "uri": str(block.uri), "name": block.name}
    if isinstance(block, EmbeddedResource):
        resource: Any = block.resource
        text = getattr(resource, "text", None)
        return {"type": "resource", "uri": str(resource.uri), "text": text}
    return {"type": getattr(block, "type", "unknown")}


def _decoded_length(data: str) -> int:
    """The payload's size in bytes, from its base64 length — nothing is decoded."""
    padding = len(data) - len(data.rstrip("="))
    return max(len(data) * 3 // 4 - padding, 0)
