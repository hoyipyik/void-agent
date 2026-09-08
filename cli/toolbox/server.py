"""The server: the capabilities mounted on one root, each closed over it,
and the stdio loop the shell talks to.

    void --serve-toolbox /path/to/repo        # what the shell runs for itself
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from cli.toolbox.files import (
    edit_text,
    list_dir,
    move,
    read_many,
    read_text,
    tree_of,
    write_text,
)
from cli.toolbox.process import RUN_TIMEOUT_SECONDS, run_in
from cli.toolbox.search import MAX_RESULTS, count_in, files_in, search_in

SERVER_NAME = "toolbox"


def build_server(root: Path) -> MCPServer:
    """The tools, each closed over the one root."""
    server = MCPServer(SERVER_NAME)

    async def search(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
        fixed: bool = False,
        hidden: bool = False,
        context: int = 0,
        max_results: int = MAX_RESULTS,
    ) -> str:
        """Search file contents with ripgrep. `pattern` is a regular
        expression unless `fixed` is true. `path` narrows to a subdirectory
        or file, `glob` to names like '*.py'. `context` adds surrounding
        lines. Results come back as path:line:text, relative to the root,
        and are capped — narrow the search rather than raising the cap."""
        return await search_in(
            root, pattern, path, glob, ignore_case, fixed, hidden, context, max_results
        )

    async def count_matches(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        ignore_case: bool = False,
        fixed: bool = False,
    ) -> str:
        """How many times a pattern matches, per file. Cheap: use it to
        size a search before reading anything."""
        return await count_in(root, pattern, path, glob, ignore_case, fixed)

    async def list_files(glob: str | None = None, path: str | None = None) -> str:
        """Every file ripgrep would search, honouring .gitignore. `glob`
        filters by name, `path` narrows to a subdirectory."""
        return await files_in(root, glob, path)

    async def read_file(path: str, head: int = 0, tail: int = 0) -> str:
        """Read a file's text. `head` returns only the first N lines and
        `tail` only the last N — use one of them on a large file."""
        return read_text(root, path, head, tail)

    async def list_directory(path: str | None = None) -> str:
        """What is in a directory: names, directories marked, sizes."""
        return list_dir(root, path)

    async def read_files(paths: list[str]) -> str:
        """Read several files at once. Cheaper than one call each when you
        are comparing or gathering."""
        return read_many(root, paths)

    async def directory_tree(path: str | None = None, depth: int = 3) -> str:
        """The shape of a directory, indented, without the contents. Use it
        to find your way before reading anything."""
        return tree_of(root, path, depth)

    async def write_file(path: str, content: str) -> str:
        """Write a file, replacing it if it exists, creating parent
        directories as needed. Stays inside the root."""
        return write_text(root, path, content)

    async def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace exact text in a file — the way to change one part of it
        without rewriting the whole. `old` must match exactly and appear
        once, unless `replace_all` is set."""
        return edit_text(root, path, old, new, replace_all)

    async def move_file(source: str, destination: str) -> str:
        """Move or rename a file. Refuses to overwrite what is already
        there."""
        return move(root, source, destination)

    async def run(
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        timeout: float = RUN_TIMEOUT_SECONDS,
    ) -> str:
        """Run a command inside the root and return its exit code and
        output. There is no shell: `args` is a list, so a pipe or a
        redirect written there is passed through as text, not interpreted.
        Use it for tests, builds, linters, git."""
        return await run_in(root, command, args, cwd, timeout)

    for capability in (
        search,
        count_matches,
        list_files,
        read_file,
        read_files,
        list_directory,
        directory_tree,
        write_file,
        edit_file,
        move_file,
        run,
    ):
        server.add_tool(capability)
    return server


def serve(root_path: str) -> None:
    """Run the server on stdio until its parent closes the pipe."""
    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    build_server(root).run()


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: toolbox <root>   — the one directory it may touch")
    serve(arguments[0])


if __name__ == "__main__":
    main()
