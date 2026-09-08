"""The app: what is the process's.

`VoidApp` holds what outlives a session — the config and the file it is
written to, the session store, the registry of mounted agents, the MCP
bench and the skills shelf, the local Ollama server, the OS clipboard —
and runs the flows that change them: `/model`, `/key`, `/agent`, `/mcp`
and `/skill`, each a modal read with the keys (`cli/screens/`), its
result saved to the config file. With no provider configured it asks for
a key first. The session itself — the log, the composer, the turn — is
the `Shell` (`cli/shell.py`), the one screen the app shows; the modals
open above it.

Mounting is the process's act, choosing the session's: the servers
`mcp.json` names start here, once, and a server switched in `/mcp`
starts or stops at once; a tool's or a skill's mark lands on the next
turn, when the registry builds the agent again from the config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App
from textual.binding import BindingType
from textual.screen import Screen

from cli.agents import REGISTRY, Registry
from cli.clipboard import Clipboard
from cli.commands import AgentPick, Command, Key, McpPick, Model, SkillPick
from cli.config import Config, Switch, ToolState, save_config
from cli.labels import model_label, tilde
from cli.mcp.bench import LOG_DIR, Bench
from cli.mcp.spec import MCP_FILE, SKILLS_DIR, ServerSpec, builtin_server, read_servers
from cli.providers.catalog import PROVIDER_LABELS, ModelInfo, Provider, as_provider, provider_of
from cli.providers.ollama import Ollama, OllamaDown, find_installed
from cli.screens import AgentPicker, KeyPrompt, McpPicker, ModelPicker, SkillPicker
from cli.session import SessionStore
from cli.shell import Shell
from cli.theme import VOID_THEME
from cli.widgets.panels import Panel, status_panel
from void_agent import Agent
from void_agent.skills import SkillFolderError, SkillInfo, read_skills

# One turn's agent, built from the current config — a fresh one per turn,
# as the server builds one per request, so `/model` takes effect at once.
BuildAgent = Callable[[Config], Agent]


def _switch(state: str) -> Switch:
    """The board's word for a server row, as the config's two-state mark."""
    return "off" if state == "off" else "on"


def _tool_state(state: str) -> ToolState:
    """The board's word for a tool row, as the config's three-state mark."""
    return "signed" if state == "signed" else "off" if state == "off" else "on"


class VoidApp(App[None]):
    TITLE = "void"
    CSS = "Screen { background: $background; }"
    BINDINGS: ClassVar[list[BindingType]] = [("ctrl+q", "quit", "Quit")]

    def __init__(
        self,
        build_agent: BuildAgent,
        *,
        store: SessionStore,
        config: Config,
        config_file: Path,
        clipboard: Clipboard | None = None,
        agents: Registry | None = None,
        ollama: Ollama | None = None,
        bench: Bench | None = None,
        mcp_file: Path | None = None,
        skills_dir: Path | None = None,
        root: Path | None = None,
    ) -> None:
        super().__init__()
        self.build_agent = build_agent
        # The agents that exist for this process: what `/agent` can choose.
        self.agents = agents or REGISTRY
        # The MCP servers this process mounted: the same tools whichever
        # agent runs, which is why the registry, not the agent, holds them.
        self.bench = bench or Bench(log_dir=config_file.parent / LOG_DIR)
        self.agents.bench = self.bench
        self._mcp_file = mcp_file or config_file.parent / MCP_FILE
        self._skills_dir = skills_dir or config_file.parent / SKILLS_DIR
        # What void's own toolbox may touch: where the shell was started.
        self._root = root or Path.cwd()
        # What `mcp.json` names, whether it is running or not: `/mcp` lists
        # every server, so a disabled one is still visible.
        self.servers: tuple[ServerSpec, ...] = ()
        self._mounting: asyncio.Task[None] | None = None
        self.store = store
        self.config = config
        self._config_file = config_file
        # Textual's own `clipboard` is its copy buffer; this one is the OS's.
        self.os_clipboard = clipboard or Clipboard()
        # The local model server, asked what it has when the picker opens.
        self.ollama = ollama or Ollama(config.ollama_host)
        self._shell: Shell | None = None

    def get_default_screen(self) -> Screen[Any]:
        # The theme before any widget: every colour is quoted from it.
        self.register_theme(VOID_THEME)
        self.theme = VOID_THEME.name
        self._shell = Shell(self.store)
        return self._shell

    @property
    def shell(self) -> Shell:
        """The session's screen; there once the app runs."""
        assert self._shell is not None, "the shell exists once the app runs"
        return self._shell

    @property
    def home(self) -> Path:
        """Where everything the CLI keeps lives."""
        return self._config_file.parent

    def agent_label(self) -> str:
        return self.agents.label(self.config.agent)

    async def on_shell_ready(self, message: Shell.Ready) -> None:
        """The log is up: read the shelf, start the servers, and ask for a
        key if there is no provider."""
        # The shelf is files: read it now, so the first turn has it.
        self._read_shelf()
        # The servers start in the background: a subprocess takes a moment
        # and the shell should not wait on it.
        self._mounting = asyncio.create_task(self._mount_mcp(), name="mcp-mount")
        if not self.config.configured():
            self.push_screen(KeyPrompt(self.config, tilde(self._config_file)), self._key_entered)

    async def on_unmount(self) -> None:
        await self.bench.close()

    # ── what is mounted: servers and skills ────────────────────────────

    async def _mount_mcp(self, *, quiet: bool = False) -> None:
        """Start the servers `mcp.json` names and `/mcp` left on, and say
        what came up. A server that will not start is a note, never the end
        of the shell."""
        self.servers = self._specs()
        running = set(self.config.running_servers(spec.name for spec in self.servers))
        wanted = [spec for spec in self.servers if spec.name in running]
        if not wanted:
            return
        if not quiet:
            await self.shell.note(
                f"mcp: starting {len(wanted)} server{'' if len(wanted) == 1 else 's'}…"
            )
        await self.bench.open(wanted)
        count = len(self.bench.catalog)
        if not quiet:
            await self.shell.note(
                f"mcp: {count} tool{'' if count == 1 else 's'} from"
                f" {len(wanted)} server{'' if len(wanted) == 1 else 's'} — /mcp"
            )
        for name, reason in self.bench.failures:
            log = self.bench.log_for(name)
            where = f" · what it said: {tilde(log)}" if log is not None and log.exists() else ""
            await self.shell.complain(f"mcp: {name} did not start — {reason}{where}")
        self._refresh_mcp_board()

    def _refresh_mcp_board(self) -> None:
        """A board open while the servers were still coming up gets what
        the bench holds now, rather than the empty list it opened on."""
        screen = self.screen if self.is_running else None
        if isinstance(screen, McpPicker):
            screen.reload(
                self.config,
                self.bench.catalog,
                self.bench.failures,
                mounting=self.bench.mounting,
            )

    def _specs(self) -> tuple[ServerSpec, ...]:
        """void's own toolbox first, then whatever `mcp.json` names —
        an entry of the same name there replaces it, so the built-in is a
        default and never a fence."""
        named = read_servers(self._mcp_file)
        builtin = builtin_server(self._root)
        if any(spec.name == builtin.name for spec in named):
            return named
        return (builtin, *named)

    def _read_shelf(self) -> tuple[SkillInfo, ...]:
        """The shelf, re-read: a skill edited or added while the shell runs
        is picked up without a restart. A folder that cannot be read as a
        skill is said so and skipped."""
        try:
            self.agents.skills = read_skills(self._skills_dir)
        except SkillFolderError as error:
            self.agents.skills = ()
            self.call_later(self.shell.complain, f"skills: {error}")
        return tuple(self.agents.skills)

    async def _remount(self) -> None:
        """Stop what is running and start what is wanted. A person can reach
        /mcp before the servers finished starting: let that mount land
        first, or the two race for the same bench. Awaiting here holds this
        handler, not the loop."""
        if self._mounting is not None and not self._mounting.done():
            await self._mounting
        await self.bench.close()
        await self._mount_mcp(quiet=True)

    # ── the commands that are the process's ────────────────────────────

    async def run_command(self, command: Command) -> None:
        """A slash command that changes the process's state; the shell has
        already run its own."""
        match command:
            case Model(name=name):
                if name is None:
                    await self._pick_model()
                else:
                    provider = provider_of(name) or self.config.provider or "anthropic"
                    if provider == "ollama":
                        await self._switch_to_ollama(name)
                    else:
                        await self._switch_model(provider, name)
            case AgentPick(name=name):
                if name is None:
                    self.push_screen(
                        AgentPicker(self.agents.entries, self.config.agent), self._agent_picked
                    )
                elif self.agents.describe(name) is None:
                    await self.shell.complain(
                        f"no agent named {name} is mounted — /agent lists them; mount your own"
                        " at start with --agent module:function"
                    )
                else:
                    await self._switch_agent(name)
            case McpPick():
                self.push_screen(
                    McpPicker(
                        self.servers,
                        self.bench.catalog,
                        self.config,
                        failures=self.bench.failures,
                        path=tilde(self._mcp_file),
                        mounting=self.bench.mounting,
                    ),
                    self._mcp_marked,
                )
            case SkillPick():
                self.push_screen(
                    SkillPicker(self._read_shelf(), self.config, path=tilde(self._skills_dir)),
                    self._skills_marked,
                )
            case Key(provider=named):
                provider = as_provider(named) if named else None
                if named and provider is None:
                    await self.shell.complain(
                        f"unknown provider {named} — anthropic, openai or ollama"
                    )
                elif provider == "ollama":
                    await self.shell.note("Ollama takes no key — pick one of its installed models")
                    await self._pick_model()
                else:
                    self.push_screen(
                        KeyPrompt(self.config, tilde(self._config_file), provider),
                        self._key_entered,
                    )
            case _:
                pass

    async def status(self) -> Panel:
        """`/status`: the process and the session, in one panel."""
        session = self.shell.session
        return status_panel(
            config=self.config,
            ollama_host=self.ollama.host,
            installed=await self._installed(),
            mcp=self.bench.summary(self.config, self.servers),
            mcp_file=tilde(self._mcp_file),
            skills=self.agents.skills_summary(self.config),
            skills_dir=tilde(self._skills_dir),
            session_title=session.title,
            messages=len(session.messages),
            home=tilde(self.home),
        )

    # ── the model and its key ──────────────────────────────────────────

    async def _installed(self) -> tuple[ModelInfo, ...] | None:
        """What Ollama has installed, or None when it does not answer."""
        try:
            return await self.ollama.installed()
        except OllamaDown:
            return None

    async def _pick_model(self) -> None:
        installed = await self._installed()
        self.push_screen(ModelPicker(self.config, self.ollama.host, installed), self._model_picked)

    async def _model_picked(self, choice: tuple[Provider, str] | None) -> None:
        if choice is not None:
            await self._switch_model(*choice)

    async def _switch_to_ollama(self, name: str) -> None:
        """A bare Ollama name, checked against what is installed — nothing
        else would catch a typo before the first turn."""
        try:
            installed = await self.ollama.installed()
        except OllamaDown as down:
            await self.shell.complain(f"{down} — start it with `ollama serve`")
            return
        found = find_installed(installed, name)
        if found is None:
            await self.shell.complain(
                f"{name} is not installed — `ollama pull {name}`, or /model lists what is"
            )
            return
        await self._switch_model("ollama", found)

    async def _switch_model(self, provider: Provider, model: str) -> None:
        self._set_config(self.config.with_model(provider, model))
        await self.shell.note(f"model: {model_label(self.config)}")
        if not self.config.configured():
            self.push_screen(
                KeyPrompt(self.config, tilde(self._config_file), provider), self._key_entered
            )

    async def _key_entered(self, result: tuple[Provider, str] | None) -> None:
        if result is None:
            await self.shell.note("no provider configured — /model picks one, /key adds a key")
            return
        provider, key = result
        if provider == "ollama":  # no key to take: the choice is a model
            await self._pick_model()
            return
        self._set_config(self.config.with_key(provider, key))
        await self.shell.note(
            f"{PROVIDER_LABELS[provider]} key saved to {tilde(self._config_file)}"
        )

    # ── the agent ──────────────────────────────────────────────────────

    async def _agent_picked(self, choice: str | None) -> None:
        if choice is not None:
            await self._switch_agent(choice)

    async def _switch_agent(self, name: str) -> None:
        self._set_config(self.config.with_agent(name))
        await self.shell.note(f"agent: {self.agent_label()}")

    # ── the marks: /mcp and /skill ─────────────────────────────────────

    async def on_mcp_picker_server_toggled(self, message: McpPicker.ServerToggled) -> None:
        """A server switched on or off in `/mcp`, applied now: it is a
        process, so it starts or stops here, and the board is handed what
        the bench holds afterwards."""
        message.stop()
        self._set_config(self.config.with_server_state(message.name, _switch(message.state)))
        await self._remount()

    async def _mcp_marked(self, states: dict[str, str] | None) -> None:
        """What `/mcp` marked, kept. A tool's mark lands on the next turn —
        the agent is rebuilt every one. A server's does not: turning one on
        or off starts or stops a process, so the bench is remounted here and
        the person is told what happened."""
        if not states:
            return
        config = self.config
        for key, state in states.items():
            kind, _, name = key.partition(":")
            # A server was already applied when its switch was thrown.
            if kind == "tool":
                config = config.with_tool_state(name, _tool_state(state))
        if config != self.config:
            self._set_config(config)
        running = len(self.config.running_servers(spec.name for spec in self.servers))
        signed, off = len(self.config.mcp_signed), len(self.config.mcp_off)
        await self.shell.note(
            f"mcp: {running} server{'' if running == 1 else 's'} running,"
            f" {len(self.bench.catalog)} tools · {signed} signed, {off} off"
        )

    async def _skills_marked(self, states: dict[str, str] | None) -> None:
        """What `/skill` marked. Nothing starts or stops — a skill is a
        folder — so this lands on the next turn."""
        if not states:
            return
        config = self.config
        for skill_id, state in states.items():
            config = config.with_skill_state(skill_id, "off" if state == "off" else "on")
        if config == self.config:
            return
        self._set_config(config)
        on = sum(1 for state in states.values() if state != "off")
        await self.shell.note(f"skills: {on} of {len(states)} on")

    def _set_config(self, config: Config) -> None:
        self.config = config
        save_config(self._config_file, config)
        self.shell.refresh_label()
