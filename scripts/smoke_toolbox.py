#!/usr/bin/env python
"""Does the packed binary's own MCP server answer?

`void --serve-toolbox <root>` is a mode of the same executable, and it is
the part a bundler is most likely to have broken: the MCP SDK resolves its
transports by name, so a missing submodule shows up here — at the first
call, in someone's terminal — rather than at build time. The release matrix
(`.github/workflows/binaries.yml`) runs this against every binary it packs.

One round trip is enough: mount it, ask what it offers, search a file we
just wrote. Usage: `python scripts/smoke_toolbox.py dist/void`.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from void_agent import EventSender
from void_agent.mcp import McpServer

WANTED = {"search", "read_file", "list_files", "run"}


async def check(binary: Path) -> int:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "note.txt").write_text("a needle here\n", encoding="utf-8")
        async with McpServer.stdio(str(binary), "--serve-toolbox", str(root)) as server:
            offered = set(server.names)
            missing = WANTED - offered
            if missing:
                print(f"smoke: {binary} offers {sorted(offered)}, missing {sorted(missing)}")
                return 1
            print(f"  {len(offered)} tools: {', '.join(sorted(offered))}")
            answer = await server.tool("search").invoke({"pattern": "needle"}, EventSender())
            if "note.txt" not in str(answer):
                print(f"smoke: search found nothing — {answer!r}")
                return 1
            print("  search found note.txt")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: smoke_toolbox.py <binary>", file=sys.stderr)
        return 2
    binary = Path(arguments[0]).resolve()
    if not binary.is_file():
        print(f"smoke: {binary} is not a file", file=sys.stderr)
        return 2
    return asyncio.run(check(binary))


if __name__ == "__main__":
    raise SystemExit(main())
