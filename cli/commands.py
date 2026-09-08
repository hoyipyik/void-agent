"""Slash commands typed into the composer. A line that starts with "/" is
a command and never reaches the model; anything else is a message.

Each command is a `Spec` — its name, the line the menu shows, the
argument it takes, its aliases. `matching` is what the menu lists as the
name is typed; `complete` rewrites what was typed into the chosen
command; `parse` reads a finished line."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Spec:
    name: str
    summary: str
    argument: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def usage(self) -> str:
        return f"/{self.name} {self.argument}" if self.argument else f"/{self.name}"


COMMANDS: tuple[Spec, ...] = (
    Spec("model", "switch the model — pick from the list, or name an id", "[id]"),
    Spec("session", "resume a saved session, or start a new one", aliases=("resume",)),
    Spec("agent", "switch among the mounted agents — pick from the list", "[name]"),
    Spec("mcp", "the MCP servers and their tools: on, signed, or off"),
    Spec("skill", "the skills on the shelf: on or off", aliases=("skills",)),
    Spec("new", "start a new session"),
    Spec("clear", "empty the current session"),
    Spec("attach", "attach a file — or drag one in, or write @path", "<path>"),
    Spec("paste", "attach the clipboard's image or copied file (ctrl+v)"),
    Spec("detach", "drop the pending attachments"),
    Spec("key", "enter or replace a provider's API key", "[provider]"),
    Spec("status", "the model, the session, where things are kept"),
    Spec("help", "commands and keys"),
    Spec("quit", "leave", aliases=("exit",)),
)

_BY_NAME: dict[str, Spec] = {
    alias: spec for spec in COMMANDS for alias in (spec.name, *spec.aliases)
}


@dataclass(frozen=True, slots=True)
class SessionPick:
    pass


@dataclass(frozen=True, slots=True)
class New:
    pass


@dataclass(frozen=True, slots=True)
class Clear:
    pass


@dataclass(frozen=True, slots=True)
class Model:
    name: str | None


@dataclass(frozen=True, slots=True)
class Attach:
    path: str


@dataclass(frozen=True, slots=True)
class Paste:
    pass


@dataclass(frozen=True, slots=True)
class Detach:
    pass


@dataclass(frozen=True, slots=True)
class AgentPick:
    name: str | None


@dataclass(frozen=True, slots=True)
class McpPick:
    pass


@dataclass(frozen=True, slots=True)
class SkillPick:
    pass


@dataclass(frozen=True, slots=True)
class Key:
    provider: str | None


@dataclass(frozen=True, slots=True)
class Status:
    pass


@dataclass(frozen=True, slots=True)
class Help:
    pass


@dataclass(frozen=True, slots=True)
class Quit:
    pass


@dataclass(frozen=True, slots=True)
class Unknown:
    name: str


Command = (
    SessionPick
    | New
    | Clear
    | Model
    | AgentPick
    | McpPick
    | SkillPick
    | Attach
    | Paste
    | Detach
    | Key
    | Status
    | Help
    | Quit
    | Unknown
)


def _split(text: str) -> tuple[str, str]:
    """The name typed after the slash, and the argument after it."""
    name, _, rest = text[1:].partition(" ")
    return name, rest.strip()


def matching(text: str) -> list[Spec]:
    """The commands whose name — or an alias — starts with what has been
    typed so far; every command for a bare "/"."""
    if not text.startswith("/") or "\n" in text:
        return []
    typed, _ = _split(text)
    typed = typed.lower()
    return [
        spec
        for spec in COMMANDS
        if any(name.startswith(typed) for name in (spec.name, *spec.aliases))
    ]


def complete(text: str, spec: Spec) -> str:
    """What was typed, with the name replaced by the chosen command's;
    the argument, if any, stays."""
    _, argument = _split(text)
    return f"/{spec.name} {argument}" if argument else f"/{spec.name}"


def parse(text: str) -> Command | None:
    if not text.startswith("/"):
        return None
    name, argument = _split(text)
    if not name:
        return None
    spec = _BY_NAME.get(name.lower())
    match spec.name if spec else None:
        case "session":
            return SessionPick()
        case "new":
            return New()
        case "clear":
            return Clear()
        case "model":
            return Model(argument or None)
        case "agent":
            return AgentPick(argument or None)
        case "mcp":
            return McpPick()
        case "skill":
            return SkillPick()
        case "attach":
            return Attach(argument)
        case "paste":
            return Paste()
        case "detach":
            return Detach()
        case "key":
            return Key(argument or None)
        case "status":
            return Status()
        case "help":
            return Help()
        case "quit":
            return Quit()
        case _:
            return Unknown(name)
