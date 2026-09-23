"""Every MCP tool mounted for this process.

Each server is kept by one task of its own. The MCP SDK opens its
transports inside anyio task groups, which must be closed by the task that
opened them — so `open` starts a keeper per server that mounts it and then
waits, and `close` asks them all to unwind. Nothing else touches a stack.

A server that dies while held — its connection cut, its process gone — is
its own keeper's business: the transport cancels that task and raises on
the way out, and the keeper takes the server's tools off the bench, writes
down what happened and says so. The other servers stay up, and nothing a
server does reaches the shell as an exception: the shell reports, it never
crashes.

A server's own noise goes to a file: a stdio server writes its log to
stderr, and in a terminal UI that is the canvas being drawn on."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from cli.config import Config
from cli.mcp.spec import SEPARATOR, Failure, ServerSpec, ToolInfo, open_server, signature_of
from void_agent import Tool
from void_agent.mcp import McpServer

# Where a server's own noise goes: one file per server, rewritten each
# mount so what is there is this run's.
LOG_DIR = "logs"

Opener = Callable[[ServerSpec], McpServer]
# Told when a server that was up goes down: its name, and what happened.
Dropped = Callable[[str, str], None]


class Bench:
    """Every MCP tool mounted for this process."""

    def __init__(
        self,
        opener: Opener | None = None,
        log_dir: Path | None = None,
        on_drop: Dropped | None = None,
    ) -> None:
        self._log_dir = log_dir
        self._opener = opener or self._open
        self.log_dir = log_dir
        self.on_drop = on_drop
        self._servers: dict[str, McpServer] = {}
        # In the order the specs came, whether or not a server is up yet:
        # the catalog reads the same however the mounts land.
        self._infos: dict[str, tuple[ToolInfo, ...]] = {}
        self._failures: tuple[Failure, ...] = ()
        self._keepers: tuple[asyncio.Task[None], ...] = ()
        self._closing = asyncio.Event()
        self._mounted: asyncio.Future[Any] | None = None

    @property
    def catalog(self) -> tuple[ToolInfo, ...]:
        return tuple(info for infos in self._infos.values() for info in infos)

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
        """The servers that are not up, and what happened: `did not start —
        …` for one that never came up, `dropped — …` for one that was up."""
        return self._failures

    def summary(self, config: Config, servers: Sequence[ServerSpec]) -> str:
        """What the bench holds, in one line: the tools mounted, and how
        many of them the person turned off or marked as signed."""
        mounted = len(self.catalog)
        if not mounted:
            if not servers:
                return "no servers"
            return "none started" if self._failures else "all off"
        ids = {info.id for info in self.catalog}
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
        if self._keepers:
            return
        # A bench that was closed is opened again when `/mcp` turns a server
        # on or off: the previous close must not end this mount at once.
        self._closing = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._infos = {spec.name: () for spec in specs}
        arrivals = [loop.create_future() for _ in specs]
        self._keepers = tuple(
            asyncio.create_task(self._keep(spec, arrival), name=f"mcp-{spec.name}")
            for spec, arrival in zip(specs, arrivals, strict=True)
        )
        self._mounted = asyncio.gather(*arrivals)
        await self._mounted

    async def close(self) -> None:
        """Ask every keeper to unwind, and wait for them. A keeper that ends
        badly — a transport that will not close — is nobody's concern on
        the way out: nothing here may stop the shell from closing."""
        keepers, self._keepers = self._keepers, ()
        self._closing.set()
        self._mounted = None
        await asyncio.gather(*keepers, return_exceptions=True)
        self._servers, self._infos, self._failures = {}, {}, ()

    def tools(
        self, *, state: Callable[[ToolInfo], str] = lambda info: info.default
    ) -> tuple[Tool, ...]:
        """The tools the agent gets, gated where `state` says. Built fresh —
        the agent is rebuilt every turn, so a mark lands on the next one."""
        built: list[Tool] = []
        for info in self.catalog:
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

    async def _keep(self, spec: ServerSpec, arrival: asyncio.Future[None]) -> None:
        """Mount one server, say so, and hold it open until asked to close —
        all on this one task, as the transports require. Whatever the
        server does ends here: recorded and told while the bench is up,
        dropped on the way out."""
        started = False
        try:
            async with self._opener(spec) as server:
                self._servers[spec.name] = server
                self._infos[spec.name] = tuple(_infos(spec, server))
                started = True
                # The opener may have stopped waiting — the app cancels its
                # mount when it closes — and a cancelled future takes no
                # result.
                if not arrival.done():
                    arrival.set_result(None)
                try:
                    await self._closing.wait()
                finally:
                    # Asked to close or thrown out, the tools go before the
                    # connection does: a turn built meanwhile must not reach
                    # a session that is gone.
                    self._servers.pop(spec.name, None)
                    self._infos[spec.name] = ()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not self._closing.is_set():
                reason = _reason(error)
                what = "dropped" if started else "did not start"
                self._failures += ((spec.name, f"{what} — {reason}"),)
                if started and self.on_drop is not None:
                    self.on_drop(spec.name, reason)
        finally:
            if not arrival.done():
                arrival.set_result(None)


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


def _reason(error: BaseException) -> str:
    """What went wrong, readable. A transport's failure arrives as an
    exception group that only says how many there were: its leaves are the
    words. One that says nothing — httpx's `ReadError` — is its name."""
    if isinstance(error, BaseExceptionGroup):
        group = cast("BaseExceptionGroup[BaseException]", error)
        return "; ".join(_reason(leaf) for leaf in group.exceptions)
    return str(error) or type(error).__name__
