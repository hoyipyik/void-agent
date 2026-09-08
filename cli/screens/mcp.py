"""`/mcp`: the servers and the tools they brought, each a switch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from textual.content import Content
from textual.message import Message
from textual.widgets.option_list import Option, OptionDoesNotExist

from cli.config import Config
from cli.mcp.spec import Failure, ServerSpec, ToolInfo
from cli.screens.switchboard import MARKS, PLAIN_CYCLE, TOOL_CYCLE, Switchboard


def _home(path: str) -> str:
    """A path as the person writes it, cut from the left when it is long —
    the end of a path is the part that says which one it is."""
    home = str(Path.home())
    shown = f"~{path[len(home) :]}" if path.startswith(home) else path
    return shown if len(shown) <= 44 else f"…{shown[-43:]}"


NO_SERVERS = (
    "No MCP servers yet. Name them in {path}, the shape every MCP client uses:\n\n"
    '  {{"mcpServers": {{"files": {{"command": "npx", "args":'
    ' ["-y", "@modelcontextprotocol/server-filesystem", "/data"]}}}}}}'
)


class McpPicker(Switchboard):
    """The MCP servers and the tools they brought. A server is on or off —
    turning one off stops it and takes its tools away. A tool is on, off,
    or signed: every call shows the person the exact arguments and runs
    only if they sign.

    Rows are keyed `server:<name>` and `tool:<id>`, which is how the app
    tells the two apart when the board closes."""

    TITLE_TEXT = "MCP"
    HINT = "↑↓ move · Enter/Space cycle · a server: on/off · a tool: on/signed/off · Esc done"

    class ServerToggled(Message):
        """A server was switched on or off. The app starts or stops it and
        hands the board back what the bench holds now — a switch whose
        effect waits for the board to close is not a switch."""

        def __init__(self, name: str, state: str) -> None:
            super().__init__()
            self.name = name
            self.state = state

    def __init__(
        self,
        servers: Sequence[ServerSpec],
        catalog: Sequence[ToolInfo],
        config: Config,
        *,
        failures: Sequence[Failure] = (),
        path: str = "~/.void/mcp.json",
        mounting: bool = False,
    ) -> None:
        super().__init__()
        self._servers = list(servers)
        self._catalog = list(catalog)
        self._failure = dict(failures)
        self._mounting = mounting
        self._config = config
        self.start(
            {
                **{
                    f"server:{spec.name}": config.server_state(spec.name) for spec in self._servers
                },
                **{
                    f"tool:{info.id}": config.tool_state(info.id, info.default)
                    for info in self._catalog
                },
            }
        )
        self.BLURB = self._blurb(path, mounting)

    def _blurb(self, path: str, mounting: bool) -> str:
        if not self._servers:
            return NO_SERVERS.format(path=path)
        if mounting:
            return (
                "Still starting — a server is a subprocess, and the first run of one"
                " fetched with npx or bunx downloads it. Reopen this board in a moment"
                " to see its tools."
            )
        return (
            "A signed tool shows you the call and runs only if you sign it — your"
            " word, never the server's. A server's tools start signed unless its"
            f' entry in {path} says "default": "on".'
        )

    def cycle_for(self, option_id: str) -> dict[str, str]:
        return PLAIN_CYCLE if option_id.startswith("server:") else TOOL_CYCLE

    def regroups(self, option_id: str) -> bool:
        """A server's rows are its tools: switching it hides or shows them."""
        return option_id.startswith("server:")

    def selected(self, option_id: str) -> None:
        super().selected(option_id)
        if not option_id.startswith("server:"):
            return
        name = option_id.partition(":")[2]
        # Applied now, so it is no longer a change the board has to report.
        self._initial[option_id] = self._states[option_id]
        self.post_message(self.ServerToggled(name, self._states[option_id]))

    def reload(
        self,
        config: Config,
        catalog: Sequence[ToolInfo],
        failures: Sequence[Failure],
        *,
        mounting: bool,
    ) -> None:
        """What the bench holds now, drawn where the person is looking. The
        switch they just threw keeps the highlight."""
        highlighted = self.choices.highlighted
        keep = (
            self.choices.get_option_at_index(highlighted).id
            if highlighted is not None and highlighted < self.choices.option_count
            else None
        )
        self._config, self._catalog = config, list(catalog)
        self._failure, self._mounting = dict(failures), mounting
        for info in self._catalog:
            key = f"tool:{info.id}"
            if key not in self._states:
                self._states[key] = config.tool_state(info.id, info.default)
                self._initial[key] = self._states[key]
        choices = self.choices
        choices.clear_options()
        choices.add_options(self.rows())
        if keep is not None:
            try:
                choices.highlighted = choices.get_option_index(keep)
            except OptionDoesNotExist:
                choices.highlighted = 0

    def rows(self) -> list[Option]:
        """Every server, and under each one that is on, its tools. A server
        that is off shows no tools — there are none to have — but its tools
        keep their marks, so switching it back on restores them."""
        rows: list[Option] = []
        for spec in self._servers:
            key = f"server:{spec.name}"
            rows.append(Option(self.render_row(key), id=key))
            if self._states[key] == "off":
                continue
            for info in self._catalog:
                if info.server == spec.name:
                    rows.append(Option(self.render_row(f"tool:{info.id}"), id=f"tool:{info.id}"))
        return rows

    def render_row(self, option_id: str) -> Content:
        kind, _, name = option_id.partition(":")
        if kind == "server":
            return self._server_row(name)
        info = next(info for info in self._catalog if info.id == name)
        return Content.from_markup(
            "    $mark  $name [$text-muted]$blurb[/]",
            mark=MARKS[self._states[option_id]],
            name=info.name.ljust(22),
            blurb=info.description.splitlines()[0][:40] if info.description else "",
        )

    def _server_row(self, name: str) -> Content:
        state = self._states[f"server:{name}"]
        tools = sum(1 for info in self._catalog if info.server == name)
        if state == "off":
            note = "off"
        elif name in self._failure:
            note = f"did not start: {self._failure[name]}"
        elif self._mounting and not tools:
            note = "starting…"
        elif not tools:
            note = "no tools"
        else:
            plural = "" if tools == 1 else "s"
            spec = next((s for s in self._servers if s.name == name), None)
            default = spec.default if spec is not None else "signed"
            note = f"{tools} tool{plural} · default {default}"
            if spec is not None and spec.note:
                note = f"{note} · {_home(spec.note)}"
        return Content.from_markup(
            "  $mark  $name [$text-muted]· $note[/]",
            mark=MARKS[state],
            name=name.ljust(20),
            note=note,
        )
