"""Slash commands typed into the composer: parsed when finished, listed
by the menu as they are typed."""

from __future__ import annotations

from cli.commands import (
    COMMANDS,
    AgentPick,
    Attach,
    Clear,
    Detach,
    Help,
    Key,
    Model,
    New,
    Paste,
    Quit,
    SessionPick,
    Status,
    Unknown,
    complete,
    matching,
    parse,
)


def test_slash_commands_parse() -> None:
    assert parse("/clear") == Clear()
    assert parse("/session") == SessionPick()
    assert parse("/resume") == SessionPick()
    assert parse("/new") == New()
    assert parse("/model") == Model(None)
    assert parse("/model claude-sonnet-5") == Model("claude-sonnet-5")
    assert parse("/agent") == AgentPick(None)
    assert parse("/agent deep") == AgentPick("deep")
    assert parse("/attach ~/shot.png") == Attach("~/shot.png")
    assert parse("/attach") == Attach("")
    assert parse("/paste") == Paste()
    assert parse("/detach") == Detach()
    assert parse("/key") == Key(None)
    assert parse("/key openai") == Key("openai")
    assert parse("/status") == Status()
    assert parse("/help") == Help()
    assert parse("/quit") == Quit()
    assert parse("/exit") == Quit()
    assert parse("/MODEL") == Model(None)  # case does not matter
    assert parse("/nope") == Unknown("nope")


def test_plain_text_is_not_a_command() -> None:
    assert parse("hello") is None
    assert parse("  /clear") is None  # a leading space keeps it a message
    assert parse("/") is None


def test_every_command_is_documented() -> None:
    assert {spec.name for spec in COMMANDS} == {
        "session",
        "new",
        "clear",
        "model",
        "agent",
        "mcp",
        "skill",
        "attach",
        "paste",
        "detach",
        "key",
        "status",
        "help",
        "quit",
    }
    assert all(spec.summary for spec in COMMANDS)


def test_the_menu_lists_what_the_typed_name_starts() -> None:
    assert [spec.name for spec in matching("/")] == [spec.name for spec in COMMANDS]
    assert [spec.name for spec in matching("/mo")] == ["model"]
    assert [spec.name for spec in matching("/res")] == ["session"]  # by its alias
    assert [spec.name for spec in matching("/mo claude")] == [
        "model"
    ]  # the argument is not the name
    assert matching("/zz") == []
    assert matching("hello") == []
    assert matching("/mo\nre") == []  # a second line is a message


def test_completing_keeps_the_argument() -> None:
    model = matching("/mo")[0]
    assert complete("/mo", model) == "/model"
    assert complete("/mo claude-sonnet-5", model) == "/model claude-sonnet-5"
    assert complete("/", model) == "/model"
