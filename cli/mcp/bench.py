"""Every MCP tool mounted for this process.

Each server is kept by one task of its own. The MCP SDK opens its
transports inside anyio task groups, which must be closed by the task that
opened them — so `open` starts a keeper per server that mounts it and then
waits, and `close` asks them all to unwind. Nothing else touches a stack.

A keeper is asked to close by itself, so one server is started or stopped
with the others left as they are: `start` and `stop` are what a switch
thrown in `/mcp` asks for, and neither costs any other server its tools.

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
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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


@dataclass(frozen=True, slots=True)
class _Keeper:
    """One server's keeper: the task holding it open, how that task is
    asked to close, and the word that it is up — or will not be."""

    task: asyncio.Task[None]
    closing: asyncio.Event
    arrival: asyncio.Future[None]


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
        # the catalog reads the same however the mounts land. A server
        # stopped keeps its place, so off and on does not reorder the tools.
        self._infos: dict[str, tuple[ToolInfo, ...]] = {}
        self._failures: tuple[Failure, ...] = ()
        self._keepers: dict[str, _Keeper] = {}

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
        return bool(self.starting)

    @property
    def starting(self) -> tuple[str, ...]:
        """The servers on their way up, by name."""
        return tuple(name for name, keeper in self._keepers.items() if not keeper.arrival.done())

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
        self._infos = {spec.name: () for spec in specs}
        await asyncio.gather(*(self._hold(spec) for spec in specs))

    async def start(self, spec: ServerSpec) -> None:
        """Start one server and read its tools, the others left as they
        are. One that will not start is recorded, as in `open`. One already
        up is left alone; one that dropped, or never came up, is started
        afresh — off and on is how a person asks for that."""
        keeper = self._keepers.get(spec.name)
        if keeper is not None and not keeper.task.done():
            return
        await self.stop(spec.name)
        await self._hold(spec)

    async def stop(self, name: str) -> None:
        """Stop one server, the others left as they are. Its tools go at
        once, before its connection does; what it said on the way out is
        nobody's concern, and a server that is off is not a failure."""
        keeper = self._keepers.get(name)
        if keeper is None:
            return
        keeper.closing.set()
        self._servers.pop(name, None)
        self._infos[name] = ()
        await asyncio.gather(keeper.task, return_exceptions=True)
        # Still in the books until it is down: a close that comes meanwhile
        # waits for it too.
        if self._keepers.get(name) is keeper:
            del self._keepers[name]
        self._failures = tuple(failure for failure in self._failures if failure[0] != name)

    async def close(self) -> None:
        """Ask every keeper to unwind, and wait for them. A keeper that ends
        badly — a transport that will not close — is nobody's concern on
        the way out: nothing here may stop the shell from closing."""
        keepers, self._keepers = self._keepers, {}
        for keeper in keepers.values():
            keeper.closing.set()
        await asyncio.gather(*(keeper.task for keeper in keepers.values()), return_exceptions=True)
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

    def _hold(self, spec: ServerSpec) -> asyncio.Future[None]:
        """A keeper for one server, and the word that it is up or will not
        be. Each has a closing of its own: asking one to unwind must not
        end the others, nor a keeper started after it."""
        closing = asyncio.Event()
        arrival: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._infos.setdefault(spec.name, ())
        task = asyncio.create_task(self._keep(spec, arrival, closing), name=f"mcp-{spec.name}")
        self._keepers[spec.name] = _Keeper(task, closing, arrival)
        return arrival

    async def _keep(
        self, spec: ServerSpec, arrival: asyncio.Future[None], closing: asyncio.Event
    ) -> None:
        """Mount one server, say so, and hold it open until asked to close —
        all on this one task, as the transports require. Whatever the
        server does ends here: recorded and told while it is wanted,
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
                    await closing.wait()
                finally:
                    # Asked to close or thrown out, the tools go before the
                    # connection does: a turn built meanwhile must not reach
                    # a session that is gone.
                    self._servers.pop(spec.name, None)
                    self._infos[spec.name] = ()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not closing.is_set():
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
