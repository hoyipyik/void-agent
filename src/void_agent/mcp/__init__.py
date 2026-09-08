"""MACHINERY, not a concept: the bridge to an MCP server.

A server's tools enter the runtime as ordinary `Tool`s. Nothing above this
package learns that a capability is remote — the loop schedules it, the
gate guards it, the parts record it, exactly as for a local function.

What MCP does not bring is the approval. The protocol has no notion of "a
person must sign this call", and a server would be the wrong place to
decide it: `approval` is declared HERE, in your code, per tool name. MCP
gives you the tools; this repo gives you the layer MCP lacks.

    async with McpServer.stdio("npx", "-y", "server-filesystem", "/data") as files:
        agent = Agent(llm, "void", "a file assistant")
        for capability in files.tools(approvals={"write_file": must_sign}):
            agent.tool(capability)

The MCP SDK's types appear only in this package, the way a provider's SDK
stays inside `providers/`. `import mcp` in these modules is the absolute
import of the SDK, never of this package.
"""

from void_agent.mcp.result import McpUnknownTool
from void_agent.mcp.server import McpServer

__all__ = ["McpServer", "McpUnknownTool"]
