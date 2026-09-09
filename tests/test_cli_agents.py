"""Which agent the CLI runs: one of the mounted ones — `universal`, plus
what `--agent` mounted at start. `/agent` chooses among them and nothing
else."""

from __future__ import annotations

from pathlib import Path

import cli.agents
import cli.agents.universal
import pytest
from cli.agents import CATALOG, NO_PROVIDER, AgentLoadError, Registry, load_builder
from cli.app import VoidApp
from cli.config import Config
from cli.session import SessionStore

from void_agent import Agent, ScriptedLlm


def test_every_catalogue_entry_builds_an_agent() -> None:
    registry = Registry()
    assert [info.id for info in registry.entries] == [info.id for info in CATALOG]
    assert len({info.id for info in CATALOG}) == len(CATALOG)
    for info in CATALOG:
        builder = registry.builder_for(info.id)
        assert builder is not None
        assert isinstance(builder(ScriptedLlm([])), Agent)
    assert registry.builder_for("nope") is None


def test_a_spec_names_a_module_and_a_function() -> None:
    assert load_builder("cli.agents") == cli.agents.build_agent  # the default function
    assert load_builder("cli.agents.universal") is cli.agents.universal.build_agent
    with pytest.raises(AgentLoadError, match="cannot import"):
        load_builder("no.such.module")
    with pytest.raises(AgentLoadError, match="no callable"):
        load_builder("cli.agents:nope")


def test_mounting_imports_at_once_and_lists_the_agent() -> None:
    registry = Registry()
    info = registry.mount("cli.agents.universal:chat")
    assert info.id == info.name == info.spec == "cli.agents.universal:chat"
    assert registry.describe(info.id) is info
    assert registry.entries[-1] is info
    assert registry.builder_for(info.id) is cli.agents.universal.chat
    assert registry.label("cli.agents.universal:chat") == "cli.agents.universal:chat"
    assert registry.label("universal") == "Universal"
    with pytest.raises(AgentLoadError, match="cannot import"):
        registry.mount("no.such:thing")
    assert registry.describe("no.such:thing") is None  # nothing half-mounted


def test_startup_takes_the_request_else_the_saved_choice_else_the_default() -> None:
    registry = Registry()
    assert registry.startup(None, "universal") == "universal"
    assert registry.startup(None, "gone") == "universal"  # a stale saved choice
    mounted = registry.startup("cli.agents.universal:chat", "universal")
    assert mounted == "cli.agents.universal:chat"
    assert registry.describe(mounted) is not None
    assert (
        registry.startup(None, "cli.agents.universal:chat") == "cli.agents.universal:chat"
    )  # saved, still here
    with pytest.raises(AgentLoadError):
        registry.startup("no.such:thing", "universal")


async def test_the_chosen_agent_runs_on_the_session_and_a_stale_choice_says_so(
    tmp_path: Path,
) -> None:
    registry = Registry()
    app = VoidApp(
        registry.build_agent,
        store=SessionStore(tmp_path / "sessions"),
        config=Config(agent="universal"),  # no provider: the scripted fallback speaks
        config_file=tmp_path / "config.json",
        agents=registry,
    )
    async with app.run_test() as pilot:
        await pilot.press("escape")  # decline the key prompt
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
        await pilot.pause()
        assert [reply.source for reply in app.shell.replies()] == [NO_PROVIDER]
        app.config = app.config.with_agent("gone")  # a saved choice that is no longer mounted
        await pilot.press(*"hi", "enter")
        await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
        await pilot.pause()
        assert "No agent named `gone` is mounted" in app.shell.replies()[-1].source


# ── the universal agent: the model, plus whatever MCP is mounted ─────────


async def test_the_universal_agent_carries_the_mounted_mcp_tools() -> None:
    from cli.mcp import Bench, ServerSpec
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    files = FakeMcp([descriptor("write_file"), descriptor("read_file", "reads a file")])
    bench = Bench(opener=lambda spec: McpServer(client=files))
    await bench.open((ServerSpec(name="files", command="npx"),))
    registry = Registry(bench=bench)
    try:
        agent = registry.build_agent(Config(agent="universal"))
        assert set(agent.tool_names) == {
            "update_plan",
            "reflect",
            "files__write_file",
            "files__read_file",
        }
        off = registry.build_agent(
            Config(agent="universal").with_tool_state("files__write_file", "off")
        )
        assert "files__write_file" not in off.tool_names
        assert "files__read_file" in off.tool_names
    finally:
        await bench.close()


async def test_every_agent_gets_the_same_mcp_tools_not_just_the_universal_one() -> None:
    from cli.mcp import Bench, ServerSpec
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    bench = Bench(opener=lambda spec: McpServer(client=FakeMcp([descriptor("write_file")])))
    await bench.open((ServerSpec(name="files", command="npx"),))
    registry = Registry(bench=bench)
    registry.mount("cli.agents.universal:chat")  # what `--agent` would have mounted at start
    try:
        agent = registry.build_agent(Config(agent="cli.agents.universal:chat"))
        assert "files__write_file" in agent.tool_names
        assert "update_plan" not in agent.tool_names  # chat's own shape, plus the mounted
    finally:
        await bench.close()


def test_without_a_bench_an_agent_is_exactly_what_its_builder_made() -> None:
    agent = Registry().build_agent(Config(agent="universal"))
    assert agent.tool_names == ("update_plan", "reflect")
