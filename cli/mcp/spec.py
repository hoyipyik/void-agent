"""A server as the config file names it, and how one is opened.

`read_servers` reads `~/.void/mcp.json` — the `mcpServers` shape every
other MCP client uses — into `ServerSpec`s; `builtin_server` is void's
own toolbox as a spec like any other; `open_server` turns a spec into the
real thing, a subprocess or an HTTP session. What a server's tools are
until the person says otherwise is decided here too: the safe direction,
`signed`, unless the person's own file vouches for the server."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from cli.config import ToolState
from void_agent import Approval
from void_agent.mcp import McpServer

MCP_FILE = "mcp.json"
# void's own server (`cli/toolbox/`), mounted by re-running this
# executable. Frozen, that is the binary; from a checkout, the interpreter
# with `-m cli`. Signed by default: it can write.
BUILTIN = "toolbox"
SERVE_TOOLBOX = "--serve-toolbox"
# What a server's tools are until the person says otherwise. Adding a
# server must not hand the model a set of ungated tools, so the safe
# direction is the default; `"default": "on"` in the file is the person
# vouching for a server, and it is theirs to write — a server's own
# `readOnlyHint` is its word, readable but never a decision.
DEFAULT_STATE: ToolState = "signed"
STATES: frozenset[str] = frozenset({"on", "off", "signed"})
# The shelf `/skill` lists, beside the config file.
SKILLS_DIR = "skills"
SEPARATOR = "__"


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """One server as the config file names it: a subprocess, or a URL."""

    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=lambda: cast("dict[str, str]", {}))
    url: str = ""
    # A hosted server's authentication: `{"Authorization": "Bearer …"}`.
    headers: Mapping[str, str] = field(default_factory=lambda: cast("dict[str, str]", {}))
    # What this server's tools are until `/mcp` says otherwise.
    default: ToolState = DEFAULT_STATE
    # One line for `/mcp`, where what a server reaches is not obvious from
    # its name — void's own says which directory it may touch.
    note: str = ""


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """One mounted tool, as the person sees it in `/mcp`."""

    server: str
    name: str
    id: str
    description: str
    default: ToolState = DEFAULT_STATE


Failure = tuple[str, str]


def read_servers(path: Path) -> tuple[ServerSpec, ...]:
    """The servers in `mcp.json` — the `mcpServers` shape every other MCP
    client uses. A file that is absent or unreadable is simply no servers;
    an entry naming neither a command nor a URL is skipped."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(data, dict):
        return ()
    document = cast("dict[str, Any]", data)
    entries = document.get("mcpServers")
    if not isinstance(entries, dict):
        return ()
    fallback = _state(document.get("default"), DEFAULT_STATE)
    return tuple(
        spec
        for name, entry in cast("dict[str, Any]", entries).items()
        if isinstance(entry, dict)
        if (spec := _spec(str(name), cast("dict[str, Any]", entry), fallback)) is not None
    )


def _state(value: Any, fallback: ToolState) -> ToolState:
    """A state the file names, or the fallback. A word that is not one of
    the three is not one of the three."""
    return cast("ToolState", value) if value in STATES else fallback


def _spec(name: str, entry: Mapping[str, Any], fallback: ToolState) -> ServerSpec | None:
    command = str(entry.get("command") or "")
    url = str(entry.get("url") or "")
    if not command and not url:
        return None
    raw_args = entry.get("args")
    args = tuple(str(a) for a in cast("list[Any]", raw_args)) if isinstance(raw_args, list) else ()
    return ServerSpec(
        name=name,
        command=command,
        args=args,
        env=_strings(entry.get("env")),
        url=url,
        headers=_strings(entry.get("headers")),
        default=_state(entry.get("default"), fallback),
    )


def _strings(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in cast("dict[str, Any]", value).items()}


def builtin_server(root: Path) -> ServerSpec:
    """The toolbox as a spec like any other, so `/mcp` lists it, marks it
    and switches it off with nothing special about it."""
    executable = sys.executable
    arguments = (
        (SERVE_TOOLBOX, str(root))
        if getattr(sys, "frozen", False)
        else ("-m", "cli", SERVE_TOOLBOX, str(root))
    )
    return ServerSpec(
        name=BUILTIN,
        command=executable,
        args=arguments,
        default="signed",
        note=str(root),
    )


def open_server(spec: ServerSpec, log_dir: Path | None = None) -> McpServer:
    """The real thing: a subprocess, or an HTTP session."""
    if spec.url:
        return McpServer.http(spec.url, headers=spec.headers or None)
    return McpServer.stdio(
        spec.command,
        *spec.args,
        env=spec.env or None,
        errlog=log_dir / f"{spec.name}.log" if log_dir is not None else None,
    )


def signature_of(info: ToolInfo) -> Approval:
    """The gate for a tool the person marked: every call is signed, in
    their own words — the server never had a say."""

    def approval(input: Any) -> str:
        return f"{info.id} is marked as needing your signature"

    return approval
