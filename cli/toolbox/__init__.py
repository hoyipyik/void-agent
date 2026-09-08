"""void's toolbox: one MCP server over one root — read, search, edit, run.

The shell ships this rather than asking the person to mount somebody
else's. Two reasons, both practical:

- The node filesystem server finds files by name and reads them whole but
  cannot search their contents, and it needs npx or bunx and a download.
  The obvious third-party ripgrep servers can search — but they take the
  directory as a *call argument*, so the model chooses it, and a search
  that can be pointed anywhere is a way to read anything the process can
  read.
- Everything here is ours, so it needs nothing but void: no node, no
  network, and in the packed binary no Python installation either.
  `ripgrep` rides along as a wheel where one exists; where it does not,
  reading and listing still work and search says so.

The root is a launch argument, resolved once, and every path the model
offers is checked against it (`root.py`). That is the point: a capability
is a tool you write, and the boundary lives in your code rather than in
the model's good judgement. The capabilities are `files.py` (read, list,
tree, write, edit, move), `search.py` (ripgrep: search, count, files) and
`process.py` (run); `server.py` mounts them on the one root.

Nothing here is gated. Whether a call must be signed is the application's
to decide (`/mcp`), and this server is mounted `signed` by default — a
server deciding its own danger is the one thing the design does not allow.

    void --serve-toolbox /path/to/repo        # what the shell runs for itself
"""

from cli.toolbox.server import SERVER_NAME, build_server, serve

__all__ = ["SERVER_NAME", "build_server", "serve"]
