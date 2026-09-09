"""Every MCP tool mounted for this process.

The servers are kept by one task of their own. The MCP SDK opens its
transports inside anyio task groups, which must be closed by the task that
opened them — so `open` starts a keeper that mounts everything and then
waits, and `close` asks it to unwind. Nothing else touches the stack.

A server's own noise goes to a file: a stdio server writes its log to
stderr, and in a terminal UI that is the canvas being drawn on."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from contextlib import AsyncExitStack
from pathlib import Path

from cli.config import Config
from cli.mcp.spec import SEPARATOR, Failure, ServerSpec, ToolInfo, open_server, signature_of
from void_agent import Tool
from void_agent.mcp import McpServer

# Where a server's own noise goes: one file per server, rewritten each
# mount so what is there is this run's.
LOG_DIR = "logs"

Opener = Callable[[ServerSpec], McpServer]


class Bench:
    """Every MCP tool mounted for this process."""

    def __init__(self, opener: Opener | None = None, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir
        self._opener = opener or self._open
        self.log_dir = log_dir
        self._servers: dict[str, McpServer] = {}
        self._catalog: tuple[ToolInfo, ...] = ()
        self._failures: tuple[Failure, ...] = ()
        self._keeper: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self._mounted: asyncio.Future[None] | None = None

    @property
    def catalog(self) -> tuple[ToolInfo, ...]:
        return self._catalog

    def _open(self, spec: ServerSpec) -> McpServer:
        return open_server(spec, self._log_dir)

    def log_for(self, name: str) -> Path | None:
        """Where that server's own words went, for a failure worth reading."""
        return None if self._log_dir is None else self._log_dir / f"{name}.log"

    @property
    def mounting(self) -> bool:
        """Whether the servers are still coming up. Starting one is a
        subprocess and, the first time, a download — long enough that the
        shell must be able to say so rather than show an empty list."""
        return self._mounted is not None and not self._mounted.done()

    @property
    def failures(self) -> tuple[Failure, ...]:
        """The servers that would not start, and what they said."""
        return self._failures

    def summary(self, config: Config, servers: Sequence[ServerSpec]) -> str:
        """What the bench holds, in one line: the tools mounted, and how
        many of them the person turned off or marked as signed."""
        mounted = len(self._catalog)
        if not mounted:
            if not servers:
                return "no servers"
            return "none started" if self._failures else "all off"
        ids = {info.id for info in self._catalog}
        marks = [
            f"{count} {word}"
            for word, count in (
                ("signed", len(ids & set(config.mcp_signed))),
                ("off", len(ids & set(config.mcp_off))),
            )
            if count
        ]
        tail = f" · {', '.join(marks)}" if marks else ""
        return f"{mounted} tool{'' if mounted == 1 else 's'}{tail}"

    async def open(self, specs: Sequence[ServerSpec]) -> None:
        """Start every server and read its tools. One that will not start is
        recorded and skipped: the rest of the bench still comes up."""
        if self._keeper is not None:
            return
        # A bench that was closed is opened again when `/mcp` turns a server
        # on or off: the previous close must not end this mount at once.
        self._closing = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._mounted = loop.create_future()
        self._keeper = asyncio.create_task(self._keep(specs), name="mcp-bench")
        await self._mounted

    async def close(self) -> None:
        """Ask the keeper to unwind, and wait for it."""
        keeper, self._keeper = self._keeper, None
        self._closing.set()
        self._failures = ()
        self._mounted = None
        if keeper is not None:
            await keeper
        self._servers, self._catalog = {}, ()

    def tools(
        self, *, state: Callable[[ToolInfo], str] = lambda info: info.default
    ) -> tuple[Tool, ...]:
        """The tools the agent gets, gated where `state` says. Built fresh —
        the agent is rebuilt every turn, so a mark lands on the next one."""
        built: list[Tool] = []
        for info in self._catalog:
            marked = state(info)
            if marked == "off":
                continue
            built.append(
                self._servers[info.server].tool(
                    info.name,
                    alias=info.id,
                    approval=signature_of(info) if marked == "signed" else None,
                )
            )
        return tuple(built)

    async def _keep(self, specs: Sequence[ServerSpec]) -> None:
        """Mount everything, say so, and hold the stack open until asked to
        close — all in this one task, as the transports require."""
        assert self._mounted is not None
        mounted = self._mounted
        async with AsyncExitStack() as stack:
            try:
                await self._mount(specs, stack)
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # pragma: no cover - the caller re-raises
                if not mounted.done():
                    mounted.set_exception(error)
                raise
            # The opener may have stopped waiting — the app cancels its
            # mount when it closes — and a cancelled future takes no result.
            if not mounted.done():
                mounted.set_result(None)
            await self._closing.wait()

    async def _mount(self, specs: Sequence[ServerSpec], stack: AsyncExitStack) -> None:
        catalog: list[ToolInfo] = []
        failures: list[Failure] = []
        for spec in specs:
            try:
                server = await stack.enter_async_context(self._opener(spec))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures.append((spec.name, str(error)))
                continue
            self._servers[spec.name] = server
            catalog.extend(_infos(spec, server))
        self._catalog, self._failures = tuple(catalog), tuple(failures)


def _infos(spec: ServerSpec, server: McpServer) -> Iterable[ToolInfo]:
    for name in server.names:
        descriptor = server.describe(name)
        yield ToolInfo(
            server=spec.name,
            name=name,
            id=f"{spec.name}{SEPARATOR}{name}",
            description=descriptor.description or descriptor.title or name,
            default=spec.default,
        )
