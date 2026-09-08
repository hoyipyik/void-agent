"""The MCP servers this process mounted, and the tools they offer.

Mounting is a process-level act, the way `--agent` is: the servers named
in `~/.void/mcp.json` are started once, their tools discovered once, and
they stay up for as long as the app runs. Choosing is a session-level act:
`/mcp` turns one on or off, or marks it as needing the person's
signature. The agent is rebuilt every turn, so a toggle takes effect on
the next message.

`spec.py` is what the file says — a server as named, and how one is
opened; `bench.py` is what runs — every server started, every tool
discovered, held by one keeper task. Which tools a server offers is its
business; whether a call needs a signature is not. The gate comes from
what the person marked here.
"""

from cli.mcp.bench import LOG_DIR, Bench
from cli.mcp.spec import (
    MCP_FILE,
    SKILLS_DIR,
    Failure,
    ServerSpec,
    ToolInfo,
    builtin_server,
    open_server,
    read_servers,
)

__all__ = [
    "LOG_DIR",
    "MCP_FILE",
    "SKILLS_DIR",
    "Bench",
    "Failure",
    "ServerSpec",
    "ToolInfo",
    "builtin_server",
    "open_server",
    "read_servers",
]
