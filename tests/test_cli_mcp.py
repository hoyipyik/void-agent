"""The CLI's MCP bench: servers mounted once for the process, their tools
enabled, disabled and signed by the person — never by the server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cli.mcp import Bench, ServerSpec, open_server, read_servers
from tests.mcp_fakes import FakeMcp, descriptor

from void_agent import Call, EventSender, Rejected, ScriptedHuman, attended
from void_agent.mcp import McpServer

FILES = descriptor("write_file")
READS = descriptor("read_file", "reads a file")


def bench_of(**servers: FakeMcp) -> Bench:
    return Bench(opener=lambda spec: McpServer(client=servers[spec.name]))


def specs(*names: str, default: str = "on") -> tuple[ServerSpec, ...]:
    """Servers as `mcp.json` would give them. Most of these tests are about
    routing rather than gating, so they vouch for their servers."""
    return tuple(
        ServerSpec(name=name, command="npx", args=("-y", name), default=default)  # pyright: ignore[reportArgumentType]
        for name in names
    )


def test_the_config_file_names_the_servers_it_mounts(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "files": {"command": "npx", "args": ["-y", "server-filesystem", "/data"]},
                    "docs": {"url": "https://example.test/mcp"},
                }
            }
        )
    )
    files, docs = read_servers(path)
    assert files == ServerSpec(
        name="files", command="npx", args=("-y", "server-filesystem", "/data")
    )
    assert docs == ServerSpec(name="docs", url="https://example.test/mcp")


def test_a_missing_or_unreadable_config_is_simply_no_servers(tmp_path: Path) -> None:
    assert read_servers(tmp_path / "absent.json") == ()
    broken = tmp_path / "mcp.json"
    broken.write_text("{ not json")
    assert read_servers(broken) == ()


def test_an_entry_with_neither_a_command_nor_a_url_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"broken": {"args": ["x"]}, "ok": {"url": "u"}}}))
    assert [spec.name for spec in read_servers(path)] == ["ok"]


async def test_the_bench_mounts_every_server_and_qualifies_their_tools() -> None:
    bench = bench_of(files=FakeMcp([FILES, READS]), docs=FakeMcp([READS]))
    await bench.open(specs("files", "docs"))
    try:
        assert [info.id for info in bench.catalog] == [
            "files__write_file",
            "files__read_file",
            "docs__read_file",
        ]
        assert bench.catalog[0].server == "files"
        assert bench.catalog[0].name == "write_file"
        assert bench.catalog[0].description == "writes a file"
    finally:
        await bench.close()


async def test_two_servers_offering_the_same_tool_both_reach_the_agent() -> None:
    files, docs = FakeMcp([READS]), FakeMcp([READS])
    bench = bench_of(files=files, docs=docs)
    await bench.open(specs("files", "docs"))
    try:
        one, two = bench.tools()
        await one.invoke({"path": "/a"}, EventSender())
        await two.invoke({"path": "/b"}, EventSender())
    finally:
        await bench.close()
    assert (one.name, two.name) == ("files__read_file", "docs__read_file")
    assert files.calls == [("read_file", {"path": "/a"})]
    assert docs.calls == [("read_file", {"path": "/b"})]


async def test_a_server_that_will_not_start_is_reported_and_the_others_still_mount() -> None:
    def opener(spec: ServerSpec) -> McpServer:
        if spec.name == "broken":
            raise OSError("npx: not found")
        return McpServer(client=FakeMcp([READS]))

    bench = Bench(opener=opener)
    await bench.open(specs("broken", "files"))
    try:
        assert [info.id for info in bench.catalog] == ["files__read_file"]
        assert bench.failures == (("broken", "npx: not found"),)
    finally:
        await bench.close()


async def test_a_disabled_tool_is_never_given_to_the_agent() -> None:
    bench = bench_of(files=FakeMcp([FILES, READS]))
    await bench.open(specs("files"))
    try:
        kept = bench.tools(state=lambda info: "off" if info.name == "write_file" else "on")
        assert [capability.name for capability in kept] == ["files__read_file"]
    finally:
        await bench.close()


async def test_a_tool_the_person_marked_signed_asks_before_it_runs() -> None:
    files = FakeMcp([FILES])
    bench = bench_of(files=files)
    await bench.open(specs("files"))
    human = ScriptedHuman([True])
    try:
        (capability,) = bench.tools(state=lambda info: "signed")
        with attended(human):
            await capability.invoke({"path": "/etc/hosts"}, EventSender())
    finally:
        await bench.close()
    [ask] = human.asked
    assert ask.kind == "approval"
    assert "files__write_file" in ask.question
    assert ask.call == Call(tool="files__write_file", input={"path": "/etc/hosts"})
    assert files.calls == [("write_file", {"path": "/etc/hosts"})]


async def test_a_signed_tool_the_person_declines_never_reaches_the_server() -> None:
    files = FakeMcp([FILES])
    bench = bench_of(files=files)
    await bench.open(specs("files"))
    try:
        (capability,) = bench.tools(state=lambda info: "signed")
        with attended(ScriptedHuman([False])), pytest.raises(Rejected):
            await capability.invoke({"path": "/etc/hosts"}, EventSender())
    finally:
        await bench.close()
    assert files.calls == []


async def test_closing_the_bench_closes_every_server() -> None:
    files, docs = FakeMcp([READS]), FakeMcp([READS])
    bench = bench_of(files=files, docs=docs)
    await bench.open(specs("files", "docs"))
    assert files.open and docs.open
    await bench.close()
    assert files.closed and docs.closed
    assert bench.catalog == ()


async def test_a_bench_with_no_servers_mounts_and_closes_quietly() -> None:
    bench = bench_of()
    await bench.open(())
    assert bench.catalog == () and bench.tools() == ()
    await bench.close()


async def test_a_bench_can_be_closed_and_opened_again_with_a_different_set() -> None:
    """`/mcp` turning a server on or off remounts the bench; the second
    mount must not walk into the first one's closing."""
    files, docs = FakeMcp([READS]), FakeMcp([READS])
    bench = bench_of(files=files, docs=docs)
    await bench.open(specs("files", "docs"))
    assert len(bench.catalog) == 2
    await bench.close()

    await bench.open(specs("docs"))
    try:
        assert [info.server for info in bench.catalog] == ["docs"]
        (capability,) = bench.tools()
        await capability.invoke({"path": "/a"}, EventSender())
    finally:
        await bench.close()
    assert docs.calls == [("read_file", {"path": "/a"})]


def test_an_http_server_may_carry_its_own_headers() -> None:
    """A hosted server — Notion's, say — is reached with a token in a
    header. The spec carries it; nothing else needs to know."""
    spec = ServerSpec(
        name="notion",
        url="https://mcp.notion.com/mcp",
        headers={"Authorization": "Bearer ntn_x"},
    )
    assert open_server(spec) is not None  # built, not connected


def test_headers_and_env_are_read_from_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "notion": {
                        "url": "https://mcp.notion.com/mcp",
                        "headers": {"Authorization": "Bearer ntn_x"},
                    },
                    "local": {"command": "npx", "env": {"TOKEN": "t"}},
                }
            }
        )
    )
    notion, local = read_servers(path)
    assert notion.headers == {"Authorization": "Bearer ntn_x"}
    assert local.env == {"TOKEN": "t"}


async def test_a_bench_says_while_it_is_still_starting() -> None:
    """Starting a server is a subprocess and a download: the shell has to
    be able to say so rather than show an empty list."""
    started = asyncio.Event()
    bench = Bench(opener=lambda spec: _slow(started, FakeMcp([READS])))
    assert not bench.mounting
    opening = asyncio.create_task(bench.open(specs("files")))
    await asyncio.sleep(0)
    assert bench.mounting
    started.set()
    await opening
    assert not bench.mounting
    assert len(bench.catalog) == 1
    await bench.close()
    assert not bench.mounting


def _slow(gate: asyncio.Event, fake: FakeMcp) -> McpServer:
    class Slow(McpServer):
        async def __aenter__(self) -> McpServer:
            await gate.wait()
            return await super().__aenter__()

    return Slow(client=fake)


def test_a_server_declares_the_state_its_tools_start_in(tmp_path: Path) -> None:
    """Adding a server must not hand the model a set of ungated tools. What
    it gets is the person's word, written in their own file — never the
    server's `readOnlyHint`."""
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "files": {"command": "npx"},
                    "notion": {"command": "npx", "default": "on"},
                    "loud": {"command": "npx", "default": "off"},
                    "odd": {"command": "npx", "default": "nonsense"},
                }
            }
        )
    )
    files, notion, loud, odd = read_servers(path)
    assert files.default == "signed"  # nothing said: the safe direction
    assert notion.default == "on"
    assert loud.default == "off"
    assert odd.default == "signed"  # a word that is not a state is not a state


def test_one_default_may_be_set_for_every_server(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "default": "on",
                "mcpServers": {
                    "files": {"command": "npx"},
                    "docs": {"command": "npx", "default": "signed"},
                },
            }
        )
    )
    files, docs = read_servers(path)
    assert (files.default, docs.default) == ("on", "signed")


async def test_a_tools_default_travels_with_it_to_the_agent() -> None:
    fake = FakeMcp([FILES, READS])
    bench = Bench(opener=lambda spec: McpServer(client=fake))
    await bench.open((ServerSpec(name="files", command="npx", default="signed"),))
    try:
        assert {info.id: info.default for info in bench.catalog} == {
            "files__write_file": "signed",
            "files__read_file": "signed",
        }
        signed = bench.tools(state=lambda info: info.default)
        with attended(ScriptedHuman([True])):
            await signed[0].invoke({"path": "/a"}, EventSender())
        allowed = bench.tools(state=lambda info: "on")
        with attended(ScriptedHuman([])):  # nobody: an ungated call still runs
            await allowed[0].invoke({"path": "/a"}, EventSender())
        assert bench.tools(state=lambda info: "off") == ()
    finally:
        await bench.close()
