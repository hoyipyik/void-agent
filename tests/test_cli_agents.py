"""Which agent the CLI runs: one of the scanned ones. The built-in shelf,
`~/.void/agents` and every `--workspace` folder are read one level deep;
each module with a `build_agent` is an agent, and every one of them can
reach every other by name."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from cli.app import VoidApp
from cli.config import Config
from cli.registry import (
    BUILTIN,
    NO_PROVIDER,
    AgentLoadError,
    Registry,
    Source,
    folder,
)
from cli.session import SessionStore

from void_agent import Agent

AGENT = '''
    """{blurb}"""
    from void_agent import Agent, Llm

    def build_agent(llm: Llm) -> Agent:
        return Agent(llm, "{name}", "{blurb}", input_type=str)
'''


def plain(name: str, blurb: str = "a plain agent") -> str:
    return textwrap.dedent(AGENT.format(name=name, blurb=blurb))


def calling(name: str, *peers: str) -> str:
    """An agent whose builder asks the pool for each peer by name."""
    tools = "".join(f'.tool(agents("{peer}"))' for peer in peers)
    return textwrap.dedent(f'''
        """calls {", ".join(peers)}"""
        from void_agent import Agent, Llm

        def build_agent(llm: Llm, agents) -> Agent:
            return Agent(llm, "{name}", "calls others", input_type=str){tools}
    ''')


def shelf(root: Path, **files: str) -> Path:
    """A folder of agents: `name` becomes name.py, `name__init` becomes
    name/__init__.py."""
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if name.endswith("__init"):
            directory = root / name.removesuffix("__init")
            directory.mkdir()
            (directory / "__init__.py").write_text(text)
        else:
            (root / f"{name}.py").write_text(text)
    return root


# ── the built-in shelf ─────────────────────────────────────────────────


def test_the_built_in_shelf_is_scanned_the_default_first() -> None:
    registry = Registry()
    assert [info.id for info in registry.entries] == ["universal", "dummy_weather", "weather"]
    assert {info.source for info in registry.entries} == {"built-in"}
    assert all(info.error is None for info in registry.entries)
    for info in registry.entries:
        assert isinstance(registry.build_agent(Config(agent=info.id)), Agent)
    assert registry.describe("nope") is None
    assert registry.label("universal") == "universal"


# ── a folder of agents ─────────────────────────────────────────────────


def test_a_folder_is_read_one_agent_per_module(tmp_path: Path) -> None:
    home = shelf(
        tmp_path / "agents",
        researcher=plain("researcher", "digs through sources"),
        writer__init=plain("writer", "writes it up"),
        helper="def search(): ...\n",  # no build_agent: a helper, not listed
        _private=plain("private"),  # underscored: skipped
        broken="def build_agent(llm:\n",  # a syntax error: listed, with the reason
        wrong="build_agent = 3\n",
        empty='def build_agent(llm):\n    """returns nothing"""\n    return None\n',
    )
    (home / "notes.txt").write_text("not python")
    (home / "shared").mkdir()  # a folder without __init__.py: a namespace, not an agent
    registry = Registry(sources=(folder(home, label="~/agents"),))
    by_id = {info.id: info for info in registry.entries}
    assert list(by_id) == ["broken", "empty", "researcher", "writer", "wrong"]
    assert by_id["researcher"].blurb == "digs through sources"
    assert by_id["writer"].blurb == "writes it up"
    assert by_id["researcher"].source == "~/agents"
    assert by_id["researcher"].error is None
    assert by_id["broken"].error is not None and "SyntaxError" in by_id["broken"].error
    assert by_id["wrong"].error == "build_agent is not callable"
    assert by_id["empty"].error == "build_agent returned NoneType, not an Agent"
    assert registry.build_agent(Config(agent="researcher")).name == "researcher"


def test_a_broken_agent_says_why_when_chosen(tmp_path: Path) -> None:
    home = shelf(tmp_path / "agents", broken="import no_such_module\n")
    registry = Registry(sources=(folder(home, label="~/agents"),))
    info = registry.describe("broken")
    assert info is not None and info.error is not None
    assert "no_such_module" in info.error


def test_an_absent_folder_is_no_agents_and_a_rescan_sees_new_files(tmp_path: Path) -> None:
    home = tmp_path / "agents"
    registry = Registry(sources=(BUILTIN, folder(home, label="~/agents")))
    assert [info.id for info in registry.entries] == ["universal", "dummy_weather", "weather"]
    shelf(home, late=plain("late"))
    registry.scan()
    assert "late" in [info.id for info in registry.entries]
    (home / "late.py").unlink()
    registry.scan()
    assert "late" not in [info.id for info in registry.entries]


def test_siblings_import_relatively(tmp_path: Path) -> None:
    home = shelf(
        tmp_path / "agents",
        tools='def label():\n    return "shared"\n',
        user=textwrap.dedent('''
            """uses a sibling"""
            from void_agent import Agent, Llm
            from . import tools

            def build_agent(llm: Llm) -> Agent:
                return Agent(llm, tools.label(), "uses a sibling")
        '''),
    )
    registry = Registry(sources=(folder(home, label="~/agents"),))
    assert registry.build_agent(Config(agent="user")).name == "shared"


# ── the same name twice ────────────────────────────────────────────────


def test_a_name_taken_earlier_is_suffixed_in_source_order(tmp_path: Path) -> None:
    home = shelf(tmp_path / "home", writer=plain("home_writer", "the home one"))
    ws1 = shelf(tmp_path / "ws1", writer=plain("ws1_writer", "the first workspace's"))
    ws2 = shelf(tmp_path / "ws2", writer=plain("ws2_writer", "the second workspace's"))
    registry = Registry(
        sources=(
            folder(home, label="~/agents"),
            folder(ws1, label="./ws1"),
            folder(ws2, label="./ws2"),
        )
    )
    rows = [(info.id, info.source, info.blurb) for info in registry.entries]
    assert rows == [
        ("writer", "~/agents", "the home one"),
        ("writer-1", "./ws1", "the first workspace's"),
        ("writer-2", "./ws2", "the second workspace's"),
    ]
    assert registry.build_agent(Config(agent="writer-1")).name == "ws1_writer"


def test_a_built_in_name_is_not_replaced_by_a_folder(tmp_path: Path) -> None:
    home = shelf(tmp_path / "agents", universal=plain("mine", "my own universal"))
    registry = Registry(sources=(BUILTIN, folder(home, label="~/agents")))
    assert [info.id for info in registry.entries][:1] == ["universal"]
    assert registry.describe("universal-1") is not None
    assert registry.build_agent(Config(agent="universal-1")).name == "mine"


# ── the pool: every loaded agent reaches every other by name ───────────


def test_a_builder_reaches_the_pool_by_name_its_own_source_first(tmp_path: Path) -> None:
    home = shelf(
        tmp_path / "home",
        writer=plain("home_writer"),
        researcher=calling("researcher", "writer"),
        far=calling("far", "writer-1"),  # the suffixed id names the workspace's, from anywhere
    )
    ws = shelf(
        tmp_path / "ws",
        writer=plain("ws_writer"),
        editor=calling("editor", "writer"),  # bare: the writer beside it, not the pool's first
        reader=calling("reader", "researcher"),  # nothing beside it: the pool's
    )
    registry = Registry(sources=(folder(home, label="~/agents"), folder(ws, label="./ws")))
    assert all(info.error is None for info in registry.entries), [
        (info.id, info.error) for info in registry.entries
    ]

    def tools_of(name: str) -> tuple[str, ...]:
        return registry.build_agent(Config(agent=name)).tool_names

    assert tools_of("researcher") == ("home_writer",)
    assert tools_of("editor") == ("ws_writer",)
    assert tools_of("far") == ("ws_writer",)
    assert tools_of("reader") == ("researcher",)


def test_a_missing_peer_and_a_cycle_are_errors_at_scan(tmp_path: Path) -> None:
    home = shelf(
        tmp_path / "agents",
        lonely=calling("lonely", "nobody"),
        a=calling("a", "b"),
        b=calling("b", "a"),
        fine=plain("fine"),
    )
    registry = Registry(sources=(folder(home, label="~/agents"),))
    by_id = {info.id: info for info in registry.entries}
    assert by_id["lonely"].error is not None
    assert "no agent named `nobody`" in by_id["lonely"].error
    assert "a, b, fine, lonely" in by_id["lonely"].error  # what there is
    assert by_id["a"].error is not None and "a cycle: a → b → a" in by_id["a"].error
    # b's builder asks for a, which is already known to be in the cycle.
    assert by_id["b"].error is not None and "`a` cannot load" in by_id["b"].error
    assert "a → b → a" in by_id["b"].error
    assert by_id["fine"].error is None


def test_a_peer_that_cannot_load_breaks_the_one_that_needs_it(tmp_path: Path) -> None:
    home = shelf(
        tmp_path / "agents",
        needy=calling("needy", "broken"),
        broken="import no_such_module\n",
    )
    registry = Registry(sources=(folder(home, label="~/agents"),))
    needy = registry.describe("needy")
    assert needy is not None and needy.error is not None
    assert "`broken` cannot load" in needy.error


async def test_the_pool_hands_out_the_mounted_tools_once(tmp_path: Path) -> None:
    from cli.mcp import Bench, ServerSpec
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    home = shelf(
        tmp_path / "agents",
        inner=textwrap.dedent('''
            """takes what the process mounted"""
            from void_agent import Agent, Llm

            def build_agent(llm: Llm, agents) -> Agent:
                agent = Agent(llm, "inner", "takes the mounted", input_type=str)
                for capability in agents.mounted:
                    agent.tool(capability)
                return agent
        '''),
        outer=calling("outer", "inner"),
    )
    bench = Bench(opener=lambda spec: McpServer(client=FakeMcp([descriptor("write_file")])))
    await bench.open((ServerSpec(name="files", command="npx"),))
    registry = Registry(sources=(folder(home, label="~/agents"),), bench=bench)
    try:
        assert all(info.error is None for info in registry.entries)
        # As the root: what it took itself, and the registry adds nothing twice.
        assert registry.build_agent(Config(agent="inner")).tool_names == ("files__write_file",)
        # As a sub-agent: the root gets the mounted set too, beside it.
        assert registry.build_agent(Config(agent="outer")).tool_names == (
            "inner",
            "files__write_file",
        )
    finally:
        await bench.close()


# ── startup ────────────────────────────────────────────────────────────


def test_startup_takes_the_request_else_the_saved_choice_else_the_default(
    tmp_path: Path,
) -> None:
    home = shelf(tmp_path / "agents", mine=plain("mine"))
    registry = Registry(sources=(BUILTIN, folder(home, label="~/agents")))
    assert registry.startup(None, "universal") == "universal"
    assert registry.startup(None, "gone") == "universal"  # a stale saved choice
    assert registry.startup(None, "mine") == "mine"  # saved, still here
    assert registry.startup("mine", "universal") == "mine"
    with pytest.raises(AgentLoadError, match="no agent named `nope`"):
        registry.startup("nope", "universal")


def test_a_folder_source_that_is_a_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "file.py").write_text("")
    with pytest.raises(AgentLoadError, match="not a folder"):
        Registry(sources=(folder(tmp_path / "file.py", label="./file.py"),))


def test_a_source_needs_a_package_or_a_path() -> None:
    with pytest.raises(ValueError):
        Source(key="x", label="x")


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


async def test_every_agent_gets_the_same_mcp_tools_not_just_the_universal_one(
    tmp_path: Path,
) -> None:
    from cli.mcp import Bench, ServerSpec
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    home = shelf(tmp_path / "agents", mine=plain("mine"))
    bench = Bench(opener=lambda spec: McpServer(client=FakeMcp([descriptor("write_file")])))
    await bench.open((ServerSpec(name="files", command="npx"),))
    registry = Registry(sources=(BUILTIN, folder(home, label="~/agents")), bench=bench)
    try:
        agent = registry.build_agent(Config(agent="mine"))
        assert "files__write_file" in agent.tool_names
        assert "update_plan" not in agent.tool_names  # its own shape, plus the mounted
    finally:
        await bench.close()


def test_without_a_bench_an_agent_is_exactly_what_its_builder_made() -> None:
    agent = Registry().build_agent(Config(agent="universal"))
    assert agent.tool_names == ("update_plan", "reflect")
