"""The entry point: `python -m cli [--agent NAME] [--workspace DIR]`, or
the `void` binary `make cli-build` packs from this file.

Everything the CLI keeps lives under `~/.void`: `config.json` (the
provider, its key, the agent, and what `/mcp` and `/skill` marked —
written by the key prompt, `/key`, `/model`, `/agent`, `/mcp` and
`/skill`), `mcp.json` (the MCP servers to mount, the shape every MCP
client uses), `agents/` (a module per agent), `skills/` (a folder per
skill) and `sessions/`. The environment's ANTHROPIC_API_KEY /
OPENAI_API_KEY win over the file. The agents are scanned right here —
the built-in shelf, `~/.void/agents`, then each `--workspace` folder
(`cli/registry.py`) — and the one to start on is `--agent`, else
`VOID_AGENT`, else the saved choice: a name that is not there is an
error at the door, a file that cannot load is a row in `/agent` with
the reason. The packed binary carries the built-in shelf's modules
(`make cli-build` names them as hidden imports).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from cli.app import VoidApp
from cli.config import load_config
from cli.labels import tilde
from cli.registry import AGENTS_DIR, BUILTIN, AgentLoadError, Registry, folder
from cli.session import SessionStore
from cli.toolbox import serve

SERVE_TOOLBOX = "--serve-toolbox"
HOME = Path(os.environ.get("VOID_HOME") or Path.home() / ".void")
CONFIG_FILE = HOME / "config.json"
SESSIONS = HOME / "sessions"


def main(argv: Sequence[str] | None = None) -> None:
    # void is its own MCP server: the shell mounts the toolbox by
    # re-running this executable, so it needs no interpreter, no checkout
    # and no download. Handled before argparse — it is a mode, not a flag.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == [SERVE_TOOLBOX]:
        if len(arguments) != 2:
            raise SystemExit(f"usage: void {SERVE_TOOLBOX} <root>")
        serve(arguments[1])
        return
    parser = argparse.ArgumentParser(prog="void", description="the void terminal UI")
    parser.add_argument(
        "--agent",
        metavar="NAME",
        help="the agent to start on, by its name in /agent; VOID_AGENT does the same",
    )
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        action="append",
        default=[],
        help=f"a folder of agents to read beside {tilde(HOME / AGENTS_DIR)}; repeatable",
    )
    parsed = parser.parse_args(arguments)
    requested = str(parsed.agent or os.environ.get("VOID_AGENT") or "") or None
    config = load_config(os.environ, CONFIG_FILE)
    home = HOME / AGENTS_DIR
    sources = [BUILTIN, folder(home, label=tilde(home), key="home")]
    for index, name in enumerate(cast("list[str]", parsed.workspace)):
        if not Path(name).is_dir():
            parser.error(f"--workspace {name}: not a folder")
        sources.append(folder(Path(name), label=name, key=f"ws{index}"))
    try:
        registry = Registry(sources=sources)
    except AgentLoadError as error:
        parser.error(str(error))
    try:
        chosen = registry.startup(requested, config.agent)
    except AgentLoadError as error:
        parser.error(f"--agent {requested}: {error}")
    VoidApp(
        registry.build_agent,
        store=SessionStore(SESSIONS),
        config=config.with_agent(chosen),
        config_file=CONFIG_FILE,
        agents=registry,
    ).run()


if __name__ == "__main__":
    main()
