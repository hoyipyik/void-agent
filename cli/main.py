"""The entry point: `python -m cli [--agent NAME]`, or the `void` binary
`make cli-build` packs from this file.

Everything the CLI keeps lives under `~/.void`: `config.json` (the
provider, its key, the agent, and what `/mcp` and `/skill` marked —
written by the key prompt, `/key`, `/model`, `/agent`, `/mcp` and
`/skill`), `mcp.json` (the MCP servers to mount, the shape every MCP
client uses), `skills/` (a folder per skill) and `sessions/`. The
environment's ANTHROPIC_API_KEY / OPENAI_API_KEY win over the file. The
agent to start on is `--agent`, else `VOID_AGENT`, else the saved choice:
a name from the registry
(`cli/agents/registry.py`), or a `module:function` mounted right here — one that
cannot load is an error at the door. The packed binary carries the
registry's modules (`make cli-build` names them as hidden imports) and
looks for a mounted module in the working directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from cli.agents import REGISTRY, AgentLoadError
from cli.app import VoidApp
from cli.config import load_config
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
    if getattr(sys, "frozen", False):
        # The packed binary: a module mounted with --agent is looked for
        # beside the person, as `python -m cli` would find it.
        sys.path.insert(0, os.getcwd())
    parser = argparse.ArgumentParser(prog="void", description="the void terminal UI")
    parser.add_argument(
        "--agent",
        metavar="NAME",
        help=(
            "the agent to start on: "
            + ", ".join(info.id for info in REGISTRY.entries)
            + ", or module:function to mount your own; VOID_AGENT does the same"
        ),
    )
    parsed = parser.parse_args(arguments)
    requested = str(parsed.agent or os.environ.get("VOID_AGENT") or "") or None
    config = load_config(os.environ, CONFIG_FILE)
    try:
        chosen = REGISTRY.startup(requested, config.agent)
    except AgentLoadError as error:
        parser.error(f"--agent {requested}: {error}")
    VoidApp(
        REGISTRY.build_agent,
        store=SessionStore(SESSIONS),
        config=config.with_agent(chosen),
        config_file=CONFIG_FILE,
        agents=REGISTRY,
    ).run()


if __name__ == "__main__":
    main()
