"""The CLI shell: messages typed into the composer run turns on a session
that lives on disk; slash commands pick, clear and switch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from cli.agents import Registry
from cli.app import BuildAgent, VoidApp
from cli.clipboard import Clipboard
from cli.config import Config
from cli.labels import model_label
from cli.providers.ollama import Ollama
from cli.screens import AgentPicker, KeyPrompt, McpPicker, ModelPicker, SessionPicker, SkillPicker
from cli.session import SessionStore
from cli.widgets import Panel, UserBubble, Welcome
from cli.widgets.welcome import LOGO_WIDTH
from tests.test_cli_ollama import down, server
from textual.containers import VerticalScroll
from textual.widgets import Input

from void_agent import Agent, EventSender, Message, ScriptedLlm, say

CONFIGURED = Config(provider="anthropic", anthropic_api_key="sk-test")


async def finished(app: VoidApp) -> None:
    """Every turn the app started has run to its end."""
    await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]


async def mounted(app: VoidApp, pilot: Any, *, seconds: float = 30.0) -> None:
    """The MCP servers are up. Mounting runs on a task of its own — a real
    subprocess takes a moment — so waiting on the workers is not enough."""
    import asyncio

    for _ in range(int(seconds / 0.05)):
        await pilot.pause()
        if app.bench.catalog or app.bench.failures:
            return
        await asyncio.sleep(0.05)


def scripted(reply: str) -> Agent:
    return Agent(ScriptedLlm([say(reply)]), "void", "test").prompt(lambda history: list(history))


OLLAMA_HOST = "http://box:11434"


def ollama_serving() -> Ollama:
    return Ollama(OLLAMA_HOST, transport=httpx.MockTransport(server))


def ollama_down() -> Ollama:
    return Ollama(OLLAMA_HOST, transport=httpx.MockTransport(down))


def make_app(
    tmp_path: Path,
    build: BuildAgent | None = None,
    config: Config = CONFIGURED,
    ollama: Ollama | None = None,
) -> VoidApp:
    def default(_config: Config) -> Agent:
        return scripted("ok")

    return VoidApp(
        build or default,
        store=SessionStore(tmp_path / "sessions"),
        config=config,
        config_file=tmp_path / "config.json",
        ollama=ollama or ollama_down(),  # never the machine's own
    )


async def test_a_typed_message_streams_the_models_reply_and_persists_the_turn(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path, lambda _config: scripted("hello from void"))
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        assert app.shell.session.history() == [
            Message.user("hi"),
            Message.assistant("hello from void"),
        ]
        assert [reply.source for reply in app.shell.replies()] == ["hello from void"]
        stored = app.store.load(app.shell.session.id)
    assert [m.role for m in stored.messages] == ["user", "assistant"]
    assert stored.messages[1].parts == [{"type": "text", "text": "hello from void"}]


async def test_the_logo_heads_the_transcript_and_stays_there(tmp_path: Path) -> None:
    """The welcome box is the transcript's header, not a splash screen:
    the first message must not take it away."""
    app = make_app(tmp_path)
    async with app.run_test(size=(LOGO_WIDTH + 8, 30)) as pilot:
        await pilot.pause()
        assert app.query(Welcome)
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        assert isinstance(log.children[0], Welcome)  # above the first message
        assert len(app.query(Welcome)) == 1
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert len(app.query(Welcome)) == 1


async def test_each_turn_sees_the_whole_session(tmp_path: Path) -> None:
    seen: list[list[Message]] = []

    def build(_config: Config) -> Agent:
        def remember(history: list[Message]) -> list[Message]:
            seen.append(list(history))
            return list(history)

        return Agent(ScriptedLlm([say("ok")]), "void", "test").prompt(remember)

    app = make_app(tmp_path, build)
    async with app.run_test() as pilot:
        await pilot.press(*"one", "enter")
        await finished(app)
        await pilot.press(*"two", "enter")
        await finished(app)
        await pilot.pause()
    assert seen[1] == [Message.user("one"), Message.assistant("ok"), Message.user("two")]


async def test_slash_clear_empties_the_session_and_the_log(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.press(*"/clear", "enter")
        await pilot.pause()
        assert app.shell.session.history() == []
        assert app.shell.replies() == []
        assert app.store.list() == []


async def test_slash_new_starts_a_fresh_session_and_keeps_the_old_one(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        first = app.shell.session.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert app.shell.session.id != first
        assert app.shell.session.history() == []
        assert app.shell.replies() == []
        assert [s.id for s in app.store.list()] == [first]


async def test_slash_session_picks_an_earlier_session_and_replays_its_log(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    earlier = store.new()
    earlier.append("user", [{"type": "text", "text": "earlier question"}])
    earlier.append("assistant", [{"type": "text", "text": "**bold** reply"}])
    store.save(earlier)

    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/session", "enter")
        await pilot.pause()
        assert isinstance(app.screen, SessionPicker)
        # the first row is "new session"; the next is the one stored session
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.shell.session.id == earlier.id
        assert [reply.source for reply in app.shell.replies()] == ["**bold** reply"]
        await pilot.press(*"again", "enter")
        await finished(app)
        await pilot.pause()
        assert app.shell.session.history()[:2] == [
            Message.user("earlier question"),
            Message.assistant("**bold** reply"),
        ]


async def test_slash_model_changes_the_model_the_next_turn_runs_on(tmp_path: Path) -> None:
    models: list[str] = []

    def build(config: Config) -> Agent:
        models.append(config.model)
        return scripted("ok")

    app = make_app(tmp_path, build)
    async with app.run_test() as pilot:
        await pilot.press(*"/model claude-sonnet-5", "enter")
        await pilot.pause()
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
    assert models == ["claude-sonnet-5"]
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["anthropic_model"] == "claude-sonnet-5"


async def test_slash_model_alone_opens_the_picker_and_a_digit_picks(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        # OpenAI heads the catalogue: Terra, then Sol
        await pilot.press("2")
        await pilot.pause()
        assert not isinstance(app.screen, ModelPicker)
        assert (app.config.provider, app.config.model) == ("openai", "gpt-5.6-sol")


async def test_the_picker_opens_on_the_model_in_use_and_the_arrows_move(tmp_path: Path) -> None:
    app = make_app(
        tmp_path,
        config=Config(
            provider="anthropic", anthropic_api_key="k", anthropic_model="claude-sonnet-5"
        ),
    )
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.config.model == "claude-haiku-4-5"


async def test_an_openai_id_switches_the_provider_and_asks_for_its_key(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/model gpt-5.4", "enter")
        await pilot.pause()
        assert (app.config.provider, app.config.model) == ("openai", "gpt-5.4")
        assert isinstance(app.screen, KeyPrompt)  # openai has no key yet
        key = app.screen.query_one("#key", Input)
        assert key.display  # the provider is known: straight to the key
        key.value = "sk-openai"
        await pilot.press("enter")
        await pilot.pause()
        assert app.config.configured()
        assert app.config.openai_api_key == "sk-openai"
        assert app.config.anthropic_api_key == "sk-test"


async def test_slash_agent_lists_the_mounted_agents_and_refuses_any_other(tmp_path: Path) -> None:
    registry = Registry()
    registry.mount("cli.agents:chat")  # what `--agent module:function` mounts at start
    app = VoidApp(
        registry.build_agent,
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        agents=registry,
        ollama=ollama_down(),
    )
    async with app.run_test() as pilot:
        await pilot.press(*"/agent", "enter")
        await pilot.pause()
        assert isinstance(app.screen, AgentPicker)
        await pilot.press("4")  # universal, weather, dummy-weather, then the mounted one
        await pilot.pause()
        assert not isinstance(app.screen, AgentPicker)
        assert app.config.agent == "cli.agents:chat"
        await pilot.press(*"/agent no.such:thing", "enter")  # not mounted: refused at once
        await pilot.pause()
        assert app.query(".error")
        assert app.config.agent == "cli.agents:chat"
        await pilot.press(*"/agent universal", "enter")
        await pilot.pause()
        assert app.config.agent == "universal"
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["agent"] == "universal"


async def test_typing_a_slash_opens_the_menu_and_enter_runs_what_it_highlights(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/mo")
        await pilot.pause()
        assert app.shell.menu.open
        assert [spec.name for spec in [app.shell.menu.chosen] if spec] == ["model"]
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        assert not app.shell.menu.open


async def test_the_arrows_move_in_the_menu_and_tab_completes(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        names = [spec.name for spec in app.shell.menu._specs.values()]  # pyright: ignore[reportPrivateUsage]
        assert names[:2] == ["model", "session"]
        await pilot.press("down", "tab")
        await pilot.pause()
        assert app.shell.composer.text == "/session"
        await pilot.press("escape")
        await pilot.pause()
        assert not app.shell.menu.open
        assert app.shell.composer.text == "/session"  # escape only closes the menu


async def test_a_modified_enter_adds_a_line_and_enter_sends(tmp_path: Path) -> None:
    seen: list[Message] = []

    def build(_config: Config) -> Agent:
        def remember(history: list[Message]) -> list[Message]:
            seen.append(history[-1])
            return list(history)

        return Agent(ScriptedLlm([say("ok")]), "void", "test").prompt(remember)

    app = make_app(tmp_path, build)
    async with app.run_test() as pilot:
        await pilot.press("a", "shift+enter", "b", "super+enter", "c", "alt+enter", "d")
        await pilot.pause()
        assert app.shell.composer.text == "a\nb\nc\nd"
        await pilot.press("enter")
        await finished(app)
    assert seen == [Message.user("a\nb\nc\nd")]


async def test_up_in_an_empty_box_rewinds_to_an_earlier_message_and_enter_replaces_it(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"one", "enter")
        await finished(app)
        await pilot.press(*"two", "enter")
        await finished(app)
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert app.shell.composer.text == "two"
        assert [b.has_class("-target") for b in app.query(UserBubble)] == [False, True]
        await pilot.press("up")
        await pilot.pause()
        assert app.shell.composer.text == "one"
        assert [b.has_class("-target") for b in app.query(UserBubble)] == [True, False]
        await pilot.press("up")  # nothing earlier: stays
        await pilot.pause()
        assert app.shell.composer.text == "one"
        app.shell.composer.load("uno")
        await pilot.press("enter")
        await finished(app)
        await pilot.pause()
        assert app.shell.session.history() == [Message.user("uno"), Message.assistant("ok")]
        assert len(app.query(UserBubble)) == 1
        assert not app.shell.composer.rewinding
    stored = app.store.load(app.shell.session.id)
    assert [m.role for m in stored.messages] == ["user", "assistant"]


async def test_escape_or_down_past_the_last_message_leaves_the_rewind(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"one", "enter")
        await finished(app)
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert app.shell.composer.rewinding
        await pilot.press("escape")
        await pilot.pause()
        assert not app.shell.composer.rewinding
        assert app.shell.composer.text == ""
        assert not any(b.has_class("-target") for b in app.query(UserBubble))
        await pilot.press("up", "down")
        await pilot.pause()
        assert not app.shell.composer.rewinding
        assert app.shell.composer.text == ""
        assert app.shell.session.history() == [Message.user("one"), Message.assistant("ok")]


async def test_the_picker_lists_what_ollama_has_installed_and_one_needs_no_key(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path, ollama=ollama_serving())
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ModelPicker)
        head = picker.choices.get_option("head-ollama").prompt
        assert OLLAMA_HOST in str(head)
        picker.choices.highlighted = picker.choices.get_option_index("gemma4:31b-mlx")
        await pilot.press("enter")
        await pilot.pause()
        assert (app.config.provider, app.config.model) == ("ollama", "gemma4:31b-mlx")
        assert app.config.configured()
        assert not isinstance(app.screen, KeyPrompt)  # a local server asks for none
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["ollama_model"] == "gemma4:31b-mlx"
    assert saved["anthropic_api_key"] == "sk-test"  # the cloud key is kept


async def test_with_ollama_down_the_picker_says_so_and_still_lists_the_rest(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ModelPicker)
        assert "not answering" in str(picker.choices.get_option("head-ollama").prompt)
        assert picker.choices.get_option_index("gpt-5.6-terra") > 0


async def test_a_bare_ollama_name_is_checked_against_what_is_installed(tmp_path: Path) -> None:
    app = make_app(tmp_path, ollama=ollama_serving())
    async with app.run_test() as pilot:
        await pilot.press(*"/model qwen3:9b", "enter")
        await pilot.pause()
        assert app.query(".error")  # not installed: nothing changed
        assert (app.config.provider, app.config.model) == ("anthropic", "claude-opus-5")
        await pilot.press(*"/model qwen3:8b", "enter")
        await pilot.pause()
        assert (app.config.provider, app.config.model) == ("ollama", "qwen3:8b")
        # on Ollama, a name without a tag is Ollama's own shorthand for :latest
        await pilot.press(*"/model tiny", "enter")
        await pilot.pause()
        assert (app.config.provider, app.config.model) == ("ollama", "tiny:latest")
        # and the alias the picker shows is taken for the full name it stands for
        await pilot.press(*"/model qwen3.8-27b-u", "enter")  # case does not matter
        await pilot.pause()
        assert app.config.model == "someone/Qwen3.8-27B-Uncensored:q8_0"
        assert model_label(app.config) == "Ollama · qwen3.8-27b-U"  # the bar says the alias


async def test_a_bare_ollama_name_with_the_server_down_is_refused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/model qwen3:8b", "enter")
        await pilot.pause()
        assert app.query(".error")
        assert app.config.provider == "anthropic"


async def test_the_key_prompt_offers_ollama_which_opens_the_picker_instead(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path, config=Config(), ollama=ollama_serving())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, KeyPrompt)
        await pilot.press("3")  # Anthropic, OpenAI, Ollama
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        app.screen.choices.highlighted = app.screen.choices.get_option_index("qwen3:8b")
        await pilot.press("enter")
        await pilot.pause()
        assert (app.config.provider, app.config.model) == ("ollama", "qwen3:8b")


async def test_slash_key_ollama_opens_the_picker_there_is_no_key_to_take(tmp_path: Path) -> None:
    app = make_app(tmp_path, ollama=ollama_serving())
    async with app.run_test() as pilot:
        await pilot.press(*"/key ollama", "enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)


async def test_slash_status_and_slash_help_leave_panels_in_the_log(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"/status", "enter")
        await pilot.press(*"/help", "enter")
        await pilot.pause()
        assert len(app.query(Panel)) == 2


async def test_without_a_key_the_app_asks_for_a_provider_then_its_key(tmp_path: Path) -> None:
    app = make_app(tmp_path, config=Config())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, KeyPrompt)
        key = app.screen.query_one("#key", Input)
        assert not key.display  # the provider comes first
        await pilot.press("enter")  # OpenAI, the first row
        await pilot.pause()
        assert key.display
        key.value = "sk-typed"
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, KeyPrompt)
        assert (app.config.provider, app.config.api_key) == ("openai", "sk-typed")
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["openai_api_key"] == "sk-typed"


async def test_an_unknown_command_is_explained_not_sent(tmp_path: Path) -> None:
    calls = 0

    def build(_config: Config) -> Agent:
        nonlocal calls
        calls += 1
        return scripted("ok")

    app = make_app(tmp_path, build)
    async with app.run_test() as pilot:
        await pilot.press(*"/nope", "enter")
        await pilot.pause()
        assert calls == 0
        assert app.query(".error")


# ── /mcp and /skill: what is mounted, and what the person left on ───────

SKILL = """---
name: PI drafting
description: how this company writes a proforma invoice
---

Always quote in USD.
"""


def mcp_app(tmp_path: Path, config: Config = CONFIGURED) -> VoidApp:
    """A shell with two MCP servers and one skill, transports in memory."""
    from cli.mcp import Bench
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    # vouched for, so its tools start on
                    "toolbox": {"command": "npx", "args": ["-y", "fs"], "default": "on"},
                    # nothing said, so its tools start signed
                    "docs": {"url": "https://example.test/mcp"},
                }
            }
        )
    )
    (tmp_path / "skills" / "pi-drafting").mkdir(parents=True)
    (tmp_path / "skills" / "pi-drafting" / "SKILL.md").write_text(SKILL, encoding="utf-8")

    fakes = {
        "toolbox": FakeMcp([descriptor("write_file"), descriptor("read_file", "reads a file")]),
        "docs": FakeMcp([descriptor("search", "searches the docs")]),
    }
    bench = Bench(opener=lambda spec: McpServer(client=fakes[spec.name]))
    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=config,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        bench=bench,
        mcp_file=tmp_path / "mcp.json",
        skills_dir=tmp_path / "skills",
    )
    return app


async def test_the_servers_in_mcp_json_are_mounted_at_start(tmp_path: Path) -> None:
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.pause()
        assert [info.id for info in app.bench.catalog] == [
            "toolbox__write_file",
            "toolbox__read_file",
            "docs__search",
        ]


async def test_slash_mcp_lists_the_servers_and_their_tools(tmp_path: Path) -> None:
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        ids = [option.id for option in picker.choices.options]
        assert ids == [
            "server:toolbox",
            "tool:toolbox__write_file",
            "tool:toolbox__read_file",
            "server:docs",
            "tool:docs__search",
        ]


async def test_slash_mcp_marks_a_tool_signed_and_keeps_it(tmp_path: Path) -> None:
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        await pilot.press("down", "enter")  # the first tool: on → signed
        await pilot.pause()
        assert isinstance(app.screen, McpPicker)  # the board stays open
        await pilot.press("escape")
        await pilot.pause()
        assert app.config.tool_state("toolbox__write_file", "on") == "signed"
        assert app.config.tool_state("toolbox__read_file", "on") == "on"  # untouched
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["mcp_signed"] == ["toolbox__write_file"]


async def test_turning_a_server_off_stops_it_and_takes_its_tools_away(tmp_path: Path) -> None:
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        await pilot.press("enter")  # the first row is the files server: on → off
        await pilot.press("escape")
        await pilot.pause()
        await finished(app)
        assert app.config.server_state("toolbox") == "off"
        # the bench was remounted: only the server left on is running
        assert [info.server for info in app.bench.catalog] == ["docs"]
        agent = app.agents.build_agent(app.config)
        assert "toolbox__write_file" not in agent.tool_names
        assert "docs__search" in agent.tool_names


async def test_a_server_left_off_is_not_started_at_all(tmp_path: Path) -> None:
    app = mcp_app(tmp_path, config=CONFIGURED.with_server_state("docs", "off"))
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.pause()
        assert [info.server for info in app.bench.catalog] == ["toolbox", "toolbox"]
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        # it is still listed, so it can be turned back on without the file
        assert "docs" in str(picker.choices.get_option("server:docs").prompt)


async def test_slash_skill_lists_the_shelf_and_turning_one_off_drops_its_tool(
    tmp_path: Path,
) -> None:
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        assert "skill__pi_drafting" in app.agents.build_agent(app.config).tool_names
        await pilot.press(*"/skill", "enter")
        await pilot.pause()
        assert isinstance(app.screen, SkillPicker)
        await pilot.press("enter")  # on → off
        await pilot.press("escape")
        await pilot.pause()
        assert app.config.skill_state("pi-drafting") == "off"
        assert "skill__pi_drafting" not in app.agents.build_agent(app.config).tool_names


async def test_the_panels_say_where_more_of_them_go(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await mounted(app, pilot)  # the built-in server, before the board opens
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        assert isinstance(app.screen, McpPicker)
        assert "mcp.json" in app.screen.BLURB
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press(*"/skill", "enter")
        await pilot.pause()
        assert isinstance(app.screen, SkillPicker)
        assert "SKILL.md" in app.screen.BLURB
        await pilot.press("escape")
        await pilot.pause()
        assert app.config.mcp_off == () and app.config.skills_off == ()


async def test_a_server_switched_off_takes_its_tools_off_the_board(tmp_path: Path) -> None:
    """The rows under a server are its tools: turning it off hides them,
    turning it back on brings them back with their marks intact."""
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        await pilot.press("down", "enter")  # mark toolbox__write_file signed
        await pilot.pause()

        await pilot.press("up", "enter")  # the files server: on → off
        await pilot.pause()
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "server:docs",
            "tool:docs__search",
        ]
        assert picker.choices.highlighted == 0  # still on the row just switched

        await pilot.press("enter")  # back on
        await pilot.pause()
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "tool:toolbox__write_file",
            "tool:toolbox__read_file",
            "server:docs",
            "tool:docs__search",
        ]
        await pilot.press("escape")
        await pilot.pause()
    assert app.config.tool_state("toolbox__write_file") == "signed"  # the mark survived
    assert app.config.server_state("toolbox") == "on"


async def test_a_server_still_starting_says_so_rather_than_showing_nothing(
    tmp_path: Path,
) -> None:
    """The empty list a person saw while bunx was downloading looked like a
    failure. It has to read as what it is."""
    import asyncio

    from cli.mcp import Bench
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": {"toolbox": {"command": "bunx", "args": ["-y", "fs"]}}})
    )
    gate = asyncio.Event()

    class Slow(McpServer):
        async def __aenter__(self) -> McpServer:
            await gate.wait()
            return await super().__aenter__()

    bench = Bench(opener=lambda spec: Slow(client=FakeMcp([descriptor("write_file")])))
    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        bench=bench,
        mcp_file=tmp_path / "mcp.json",
        skills_dir=tmp_path / "skills",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        assert "starting…" in str(picker.choices.get_option("server:toolbox").prompt)
        assert "Still starting" in picker.BLURB
        await pilot.press("escape")
        gate.set()
        await finished(app)
        await pilot.pause()

        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        assert "1 tool" in str(picker.choices.get_option("server:toolbox").prompt)
        await pilot.press("escape")
        await pilot.pause()


async def test_a_server_nobody_vouched_for_starts_its_tools_signed(tmp_path: Path) -> None:
    """Adding a server hands the model no ungated tools: what it gets is
    what the person's own file said, and saying nothing means signed."""
    app = mcp_app(tmp_path)
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.pause()
        # no marks at all — these are the servers' own defaults
        assert app.config.mcp_on == () and app.config.mcp_signed == ()
        assert app.config.tool_state("docs__search", "signed") == "signed"

        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        assert "default on" in str(picker.choices.get_option("server:toolbox").prompt)
        assert "default signed" in str(picker.choices.get_option("server:docs").prompt)
        assert "sign" in str(picker.choices.get_option("tool:docs__search").prompt)
        assert "on" in str(picker.choices.get_option("tool:toolbox__read_file").prompt)
        agent = app.agents.build_agent(app.config)
        assert "docs__search" in agent.tool_names  # signed is mounted, not withheld
        await pilot.press("escape")
        await pilot.pause()
        # the board wrote nothing: the defaults are still the file's to change
        assert app.config.mcp_on == () and app.config.mcp_signed == ()


async def test_switching_a_server_on_brings_its_tools_in_without_closing_the_board(
    tmp_path: Path,
) -> None:
    """A switch that only takes effect after you close and reopen the board
    is not a switch. Turning a server on starts it and its tools appear
    where you are looking."""
    app = mcp_app(tmp_path, config=CONFIGURED.with_server_state("docs", "off"))
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "tool:toolbox__write_file",
            "tool:toolbox__read_file",
            "server:docs",
        ]

        await pilot.press("down", "down", "down", "enter")  # docs: off → on
        await finished(app)
        await pilot.pause()
        assert app.config.server_state("docs") == "on"
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "tool:toolbox__write_file",
            "tool:toolbox__read_file",
            "server:docs",
            "tool:docs__search",
        ]
        assert picker.choices.highlighted == 3  # still on the switch just thrown

        await pilot.press("enter")  # and back off, in place
        await finished(app)
        await pilot.pause()
        assert app.config.server_state("docs") == "off"
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "tool:toolbox__write_file",
            "tool:toolbox__read_file",
            "server:docs",
        ]
        await pilot.press("escape")
        await pilot.pause()
    assert app.config.server_state("docs") == "off"


async def test_a_board_open_while_the_servers_start_fills_in_by_itself(
    tmp_path: Path,
) -> None:
    """Opening /mcp during the first mount showed an empty list that never
    filled. It fills where the person is looking."""
    import asyncio

    from cli.mcp import Bench
    from tests.mcp_fakes import FakeMcp, descriptor

    from void_agent.mcp import McpServer

    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": {"toolbox": {"command": "bunx", "args": ["-y", "fs"]}}})
    )
    gate = asyncio.Event()

    class Slow(McpServer):
        async def __aenter__(self) -> McpServer:
            await gate.wait()
            return await super().__aenter__()

    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        bench=Bench(opener=lambda spec: Slow(client=FakeMcp([descriptor("write_file")]))),
        mcp_file=tmp_path / "mcp.json",
        skills_dir=tmp_path / "skills",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        assert [option.id for option in picker.choices.options] == ["server:toolbox"]

        gate.set()  # the server finishes coming up, board still open
        await finished(app)
        await pilot.pause()
        assert [option.id for option in picker.choices.options] == [
            "server:toolbox",
            "tool:toolbox__write_file",
        ]
        assert "1 tool" in str(picker.choices.get_option("server:toolbox").prompt)
        await pilot.press("escape")
        await pilot.pause()


async def test_a_selection_in_the_log_is_copied_to_the_os_clipboard(tmp_path: Path) -> None:
    """A drag selects in the log; ctrl+c copies. Textual's own copy is an
    OSC 52 escape, which macOS Terminal ignores and iTerm2 refuses by
    default, so the app writes the OS clipboard too. The log takes no
    focus: after the drag the composer still has the keys."""
    written: list[tuple[str, str]] = []

    def record(command: list[str], data: bytes) -> bool:
        written.append((command[0], data.decode("utf-8")))
        return True

    app = VoidApp(
        lambda _config: scripted("hello from void, worth copying"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        clipboard=Clipboard(writer=record, platform="darwin"),
        ollama=ollama_down(),
    )
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        reply = app.shell.replies()[0]
        await pilot.mouse_down(reply, offset=(0, 0))
        await pilot.hover(reply, offset=(12, 0))
        await pilot.mouse_up(reply, offset=(12, 0))
        await pilot.pause()
        assert app.focused is app.shell.composer
        await pilot.press("ctrl+c")
        await finished(app)  # the write is a worker: a subprocess, off the loop
        await pilot.pause()
        assert written == [("pbcopy", "hello from vo")]
        assert app.shell.status.line == "copied"
        await pilot.press(*"and on", "enter")  # the keys never left the composer
        await finished(app)
        await pilot.pause()
        assert app.shell.session.history()[-2] == Message.user("and on")


async def test_the_welcome_box_follows_the_model_and_the_agent(tmp_path: Path) -> None:
    """The header says which agent and model the next turn runs on — as
    the config stands now, not as it stood when the box was drawn."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        welcome = app.query_one(Welcome)
        assert model_label(CONFIGURED) in welcome.details
        was = app.agent_label()
        assert was in welcome.details
        await pilot.press(*"/model claude-sonnet-5", "enter")
        await pilot.pause()
        assert "claude-sonnet-5" in welcome.details
        await pilot.press(*"/agent weather", "enter")
        await pilot.pause()
        assert app.config.agent == "weather" and app.agent_label() != was
        assert app.agent_label() in welcome.details


async def test_the_welcome_box_drops_its_key_warning_once_a_key_is_saved(tmp_path: Path) -> None:
    app = make_app(tmp_path, config=Config())
    async with app.run_test() as pilot:
        await pilot.pause()
        welcome = app.query_one(Welcome)
        assert "no API key yet" in welcome.details
        assert isinstance(app.screen, KeyPrompt)
        await pilot.press("enter")  # OpenAI, the first row
        await pilot.pause()
        app.screen.query_one("#key", Input).value = "sk-typed"
        await pilot.press("enter")
        await pilot.pause()
        assert "no API key yet" not in welcome.details
        assert "OpenAI" in welcome.details


async def test_a_resumed_session_keeps_its_header_too(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await finished(app)
        session_id = app.shell.session.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        await app.shell.reopen(session_id)
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        assert isinstance(log.children[0], Welcome)
        assert app.shell.replies()  # the reply was replayed under it


async def test_void_mounts_its_own_toolbox_with_nothing_installed(tmp_path: Path) -> None:
    """No node, no download, no mcp.json: the shell brings read, list,
    search and write for the directory it was started in."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "note.txt").write_text("a needle here\n")
    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        mcp_file=tmp_path / "absent.json",
        skills_dir=tmp_path / "skills",
        root=work,
    )
    async with app.run_test() as pilot:
        await mounted(app, pilot)
        assert [spec.name for spec in app.servers] == ["toolbox"]
        assert app.servers[0].default == "signed"  # it can write
        assert app.bench.failures == ()
        names = {info.name for info in app.bench.catalog}
        assert names == {
            "search",
            "count_matches",
            "list_files",
            "read_file",
            "read_files",
            "list_directory",
            "directory_tree",
            "write_file",
            "edit_file",
            "move_file",
            "run",
        }

        found = app.bench.tools(state=lambda info: "on")
        search = next(t for t in found if t.name == "toolbox__search")
        answer = await search.invoke({"pattern": "needle"}, EventSender())
        assert "note.txt" in answer


async def test_an_mcp_json_entry_of_the_same_name_replaces_the_built_in(tmp_path: Path) -> None:
    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": {"toolbox": {"command": "somebody-elses", "default": "on"}}})
    )
    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        mcp_file=tmp_path / "mcp.json",
        skills_dir=tmp_path / "skills",
        root=tmp_path,
    )
    async with app.run_test() as pilot:
        await finished(app)
        await pilot.pause()
        assert [spec.name for spec in app.servers] == ["toolbox"]
        assert app.servers[0].command == "somebody-elses"  # the person's, not ours


async def test_the_board_says_which_directory_the_built_in_server_may_touch(
    tmp_path: Path,
) -> None:
    """Where void's own files server is rooted is the one thing about it
    that is not obvious, so the row says it."""
    work = tmp_path / "work"
    work.mkdir()
    app = VoidApp(
        lambda _config: scripted("ok"),
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        ollama=ollama_down(),
        mcp_file=tmp_path / "absent.json",
        skills_dir=tmp_path / "skills",
        root=work,
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await mounted(app, pilot)
        await pilot.press(*"/mcp", "enter")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, McpPicker)
        row = str(picker.choices.get_option("server:toolbox").prompt)
        assert "11 tools" in row and "default signed" in row
        assert "work" in row  # the directory it is rooted in
        await pilot.press("escape")
        await pilot.pause()
