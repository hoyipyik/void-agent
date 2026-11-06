"""A suite as its files say it.

The config names the agent, the model, the servers and the personas; the
cases files it lists hold the questions. Paths are relative to the file
that names them. Nothing here opens a connection or reads a secret: a
server's address and headers stay as written, `${NAME}` and all, and are
expanded only when a persona is mounted (`expand`), so a token that is
not set fails the cases that need it and no others.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

Outcome = Literal["answered", "asked", "failed", "cancelled", "error"]
OUTCOMES: frozenset[str] = frozenset({"answered", "asked", "failed", "cancelled", "error"})
# A case id names its record's file.
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
# A turn at the most effort may take minutes; one that takes half an hour
# is stuck.
TURN_SECONDS = 1800.0
# The config's own folder, for a server that is a program beside it.
SUITE_DIR = "${SUITE_DIR}"
# Who the person is when the suite does not say: anyone, of no trade.
PERSON_ROLE = "a person who uses a chat assistant"


class SuiteError(Exception):
    """The suite's files say something that cannot run, in words a person
    can fix the file by."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    provider: str
    model: str
    # The provider's own level; empty asks for nothing.
    effort: str = ""
    # Merged into every request, beside the effort.
    extra: Mapping[str, Any] = field(default_factory=lambda: cast("dict[str, Any]", {}))


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """One MCP server: a URL, or a command run as a subprocess."""

    name: str
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=lambda: cast("dict[str, str]", {}))
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=lambda: cast("dict[str, str]", {}))
    # Seconds to mount, and for each call after it; None leaves the SDK's.
    timeout: float | None = None
    # Tools whose every call is put to the person (glob patterns).
    signed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Persona:
    """Who a question is asked as: the servers with their credentials, and
    which of the tools they list the agent is offered."""

    name: str
    description: str = ""
    servers: Mapping[str, ServerSpec] = field(
        default_factory=lambda: cast("dict[str, ServerSpec]", {})
    )
    # Glob patterns over tool names; an empty `only` keeps every tool.
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Context:
    """What a later turn reads of an earlier one, as a server would hand it."""

    # Characters of one tool result; None for all of it.
    tool_output_limit: int | None = None
    # The tools whose calls a later turn reads back: every one, the ones
    # the servers brought, or these by name.
    tools: tuple[str, ...] | Literal["all", "mounted"] = "all"


@dataclass(frozen=True, slots=True)
class Turn:
    text: str
    attach: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class Expect:
    """What a case expects, in terms a run can check by itself."""

    outcome: Outcome | None = None
    # Every one of these tools is called.
    uses: tuple[str, ...] = ()
    # At least one of these is.
    uses_any: tuple[str, ...] = ()
    # None of these is.
    avoids: tuple[str, ...] = ()
    # No more calls than this, of any tool; 0 for an answer that needs none.
    max_calls: int | None = None
    # Tools whose calls `max_calls` does not count: the suite's, for every case.
    uncounted: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    persona: str
    turns: tuple[Turn, ...]
    # Answers to the questions the run puts to the person, in order: words
    # for the model's, a bool for a signature. Without one, a question ends
    # the turn and the next turn is the answer.
    replies: tuple[str | bool, ...] = ()
    tags: tuple[str, ...] = ()
    # What a good answer does, for whoever reads the run.
    intent: str = ""
    # What the agent cannot do yet that this question needs.
    gap: str = ""
    expect: Expect = Expect()
    # Who asks, and what they want and know, which the agent is never shown:
    # with it, the suite's person answers what the agent asks them.
    brief: str = ""


@dataclass(frozen=True, slots=True)
class Limits:
    # One turn's seconds before it is cancelled; None for no limit.
    turn_seconds: float | None = TURN_SECONDS


@dataclass(frozen=True, slots=True)
class PersonSpec:
    """The model that plays the person a case is asked by."""

    model: ModelSpec
    # Replies one case may give to what the agent asks, at most.
    rounds: int = 3
    # Who they are, in every case: "You play <role>." Each case's brief
    # says what they want and know.
    role: str = PERSON_ROLE


@dataclass(frozen=True, slots=True)
class Suite:
    name: str
    config: Path
    # `module:attribute` or `path/to/file.py:attribute`: a function of the
    # model and the mounted tools that returns the agent.
    agent: str
    paths: tuple[Path, ...]
    model: ModelSpec
    personas: Mapping[str, Persona]
    cases: tuple[Case, ...]
    runs: Path
    context: Context = Context()
    limits: Limits = Limits()
    person: PersonSpec | None = None


def load_suite(config: Path) -> Suite:
    config = config.resolve()
    document = _read(config)
    root = config.parent
    servers = {
        name: _beside(_server(name, _table(entry, f"servers.{name}")), root)
        for name, entry in _table(document.get("servers", {}), "servers").items()
    }
    personas = {
        name: _persona(name, _table(entry, f"personas.{name}"), servers)
        for name, entry in _table(document.get("personas", {}), "personas").items()
    } or {"default": Persona("default", servers=servers)}
    expect = _table(document.get("expect", {}), "expect")
    uncounted = _strings(expect.get("uncounted", []), "expect.uncounted")
    cases: list[Case] = []
    for name in _strings(document.get("cases", []), "cases"):
        cases.extend(_cases(root / name, personas, uncounted))
    _unique(cases)
    return Suite(
        name=_text(document.get("name", config.stem), "name"),
        config=config,
        agent=_agent(_required_text(document, "agent", "the config"), root),
        paths=tuple(root / path for path in _strings(document.get("paths", []), "paths")),
        model=_model(_table(document.get("model"), "model")),
        personas=personas,
        cases=tuple(cases),
        runs=root / _text(document.get("runs", "runs"), "runs"),
        context=_context(_table(document.get("context", {}), "context")),
        limits=_limits(_table(document.get("limits", {}), "limits")),
        person=_person_spec(document.get("person")),
    )


def select(
    cases: Sequence[Case],
    *,
    ids: Sequence[str] = (),
    personas: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> tuple[Case, ...]:
    """The cases every given filter keeps: an id matching one of `ids`
    (globs), one of `personas`, any one of `tags`."""
    return tuple(
        case
        for case in cases
        if (not ids or any(fnmatch.fnmatchcase(case.id, pattern) for pattern in ids))
        if not personas or case.persona in personas
        if not tags or any(tag in case.tags for tag in tags)
    )


def expand(value: str, env: Mapping[str, str], persona: str) -> str:
    """`value` with each `${NAME}` (or `${NAME:-default}`) read from `env`."""

    def variable(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        found = env.get(name) or default
        if found is None:
            raise SuiteError(f"`${{{name}}}` is not set, and persona `{persona}` needs it")
        return found

    return _VARIABLE.sub(variable, value)


def environment() -> dict[str, str]:
    return dict(os.environ)


def matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


# ── reading ───────────────────────────────────────────────────────────────


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except OSError as error:
        raise SuiteError(f"cannot read {path}: {error.strerror}") from error
    except tomllib.TOMLDecodeError as error:
        raise SuiteError(f"{path} is not TOML: {error}") from error


def _agent(spec: str, root: Path) -> str:
    """A builder named by its file is found beside the config."""
    target, colon, attribute = spec.rpartition(":")
    if colon and target.endswith(".py"):
        return f"{root / target}:{attribute}"
    return spec


def _model(table: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        provider=_required_text(table, "provider", "[model]"),
        model=_required_text(table, "model", "[model]"),
        effort=_text(table.get("effort", ""), "model.effort"),
        extra=_table(table.get("extra", {}), "model.extra"),
    )


def _person_spec(value: Any) -> PersonSpec | None:
    if value is None:
        return None
    table = _table(value, "person")
    rounds = _count(table.get("rounds", 3), "person.rounds")
    role = _text(table.get("role", ""), "person.role").strip() or PERSON_ROLE
    return PersonSpec(model=_model(table), rounds=3 if rounds is None else rounds, role=role)


def _server(name: str, table: dict[str, Any]) -> ServerSpec:
    server = ServerSpec(
        name=name,
        url=_text(table.get("url", ""), f"servers.{name}.url"),
        headers=_mapping(table.get("headers", {}), f"servers.{name}.headers"),
        command=_text(table.get("command", ""), f"servers.{name}.command"),
        args=_strings(table.get("args", []), f"servers.{name}.args"),
        env=_mapping(table.get("env", {}), f"servers.{name}.env"),
        timeout=_seconds(table.get("timeout"), f"servers.{name}.timeout"),
        signed=_strings(table.get("signed", []), f"servers.{name}.signed"),
    )
    if not server.url and not server.command:
        raise SuiteError(f"server `{name}` names neither a url nor a command")
    return server


def _beside(spec: ServerSpec, root: Path) -> ServerSpec:
    """`${SUITE_DIR}` is the config's own folder, known as it is read."""

    def here(value: str) -> str:
        return value.replace(SUITE_DIR, str(root))

    return replace(
        spec,
        url=here(spec.url),
        command=here(spec.command),
        args=tuple(here(arg) for arg in spec.args),
        env={key: here(value) for key, value in spec.env.items()},
    )


def _persona(name: str, table: dict[str, Any], servers: Mapping[str, ServerSpec]) -> Persona:
    own = _table(table.get("servers", {}), f"personas.{name}.servers")
    for server in own:
        if server not in servers:
            raise SuiteError(f"persona `{name}` sets server `{server}`, which the suite lacks")
    return Persona(
        name=name,
        description=_text(table.get("description", ""), f"personas.{name}.description"),
        servers={
            server: _overridden(spec, _table(own.get(server, {}), f"personas.{name}.{server}"))
            for server, spec in servers.items()
        },
        only=_strings(table.get("only", []), f"personas.{name}.only"),
        exclude=_strings(table.get("exclude", []), f"personas.{name}.exclude"),
    )


def _overridden(spec: ServerSpec, table: dict[str, Any]) -> ServerSpec:
    """The suite's server as a persona sets it: its headers and env added
    to the suite's, anything else replaced."""
    where = f"servers.{spec.name}"
    return replace(
        spec,
        url=_text(table.get("url", spec.url), f"{where}.url"),
        headers={**spec.headers, **_mapping(table.get("headers", {}), f"{where}.headers")},
        env={**spec.env, **_mapping(table.get("env", {}), f"{where}.env")},
        timeout=_seconds(table.get("timeout", spec.timeout), f"{where}.timeout"),
    )


def _context(table: dict[str, Any]) -> Context:
    limit = table.get("tool_output_limit")
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise SuiteError("context.tool_output_limit is a number of characters")
    tools = table.get("tools", "all")
    if isinstance(tools, str) and tools not in ("all", "mounted"):
        raise SuiteError('context.tools is "all", "mounted" or a list of tool names')
    return Context(
        tool_output_limit=limit or None,
        tools=tools if tools in ("all", "mounted") else _strings(tools, "context.tools"),
    )


def _limits(table: dict[str, Any]) -> Limits:
    return Limits(
        turn_seconds=_seconds(table.get("turn_seconds", TURN_SECONDS), "limits.turn_seconds")
    )


def _cases(path: Path, personas: Mapping[str, Persona], uncounted: tuple[str, ...]) -> list[Case]:
    document = _read(path)
    persona = _text(document.get("persona", ""), f"{path.name}: persona")
    tags = _strings(document.get("tags", []), f"{path.name}: tags")
    entries = document.get("cases", [])
    if not isinstance(entries, list):
        raise SuiteError(f"{path.name}: `cases` is a list of [[cases]] tables")
    return [
        _case(
            _table(entry, f"{path.name}: a case"), path.parent, persona, tags, personas, uncounted
        )
        for entry in cast("list[Any]", entries)
    ]


def _case(
    table: dict[str, Any],
    root: Path,
    persona: str,
    tags: tuple[str, ...],
    personas: Mapping[str, Persona],
    uncounted: tuple[str, ...],
) -> Case:
    id = _required_text(table, "id", "a case")
    if not _ID.fullmatch(id):
        raise SuiteError(f"case id `{id}` is letters, digits, `.`, `_` and `-`")
    where = f"case `{id}`"
    persona = _text(table.get("persona", persona), f"{where}: persona")
    if not persona and len(personas) == 1:
        persona = next(iter(personas))
    if persona not in personas:
        raise SuiteError(f"{where} runs as persona `{persona}`, which the suite lacks")
    turns = _turns(table, root, where)
    if not turns:
        raise SuiteError(f"{where} has nothing to ask: give it `ask` or `turns`")
    return Case(
        id=id,
        persona=persona,
        turns=turns,
        replies=_replies(table.get("replies", []), where),
        tags=tags + _strings(table.get("tags", []), f"{where}: tags"),
        intent=_text(table.get("intent", ""), f"{where}: intent"),
        gap=_text(table.get("gap", ""), f"{where}: gap"),
        expect=replace(
            _expect(_table(table.get("expect", {}), f"{where}: expect"), where),
            uncounted=uncounted,
        ),
        brief=_text(table.get("brief", ""), f"{where}: brief"),
    )


def _turns(table: dict[str, Any], root: Path, where: str) -> tuple[Turn, ...]:
    attach = tuple(root / name for name in _strings(table.get("attach", []), f"{where}: attach"))
    if "ask" in table:
        return (Turn(_text(table["ask"], f"{where}: ask"), attach),)
    raw = table.get("turns", [])
    if not isinstance(raw, list):
        raise SuiteError(f"{where}: `turns` is a list")
    turns: list[Turn] = []
    for entry in cast("list[Any]", raw):
        if isinstance(entry, str):
            turns.append(Turn(entry))
        else:
            turn = _table(entry, f"{where}: a turn")
            files = _strings(turn.get("attach", []), f"{where}: attach")
            text = _required_text(turn, "ask", f"{where}: a turn")
            turns.append(Turn(text, tuple(root / name for name in files)))
    if turns and attach:
        turns[0] = replace(turns[0], attach=attach + turns[0].attach)
    return tuple(turns)


def _replies(value: Any, where: str) -> tuple[str | bool, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str | bool) for item in cast("list[Any]", value)
    ):
        raise SuiteError(f"{where}: `replies` is a list of words and true/false")
    return tuple(cast("list[str | bool]", value))


def _expect(table: dict[str, Any], where: str) -> Expect:
    outcome = table.get("outcome")
    if outcome is not None and outcome not in OUTCOMES:
        raise SuiteError(f"{where}: expect.outcome is one of {', '.join(sorted(OUTCOMES))}")
    return Expect(
        outcome=cast("Outcome | None", outcome),
        uses=_strings(table.get("uses", []), f"{where}: expect.uses"),
        uses_any=_strings(table.get("uses_any", []), f"{where}: expect.uses_any"),
        avoids=_strings(table.get("avoids", []), f"{where}: expect.avoids"),
        max_calls=_count(table.get("max_calls"), f"{where}: expect.max_calls"),
    )


def _unique(cases: Sequence[Case]) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SuiteError(f"two cases are called `{case.id}`")
        seen.add(case.id)


# ── shapes ────────────────────────────────────────────────────────────────


def _table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SuiteError(f"`{where}` is a table")
    return cast("dict[str, Any]", value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise SuiteError(f"`{where}` is text")
    return value


def _required_text(table: dict[str, Any], key: str, where: str) -> str:
    value = _text(table.get(key, ""), f"{where}: {key}").strip()
    if not value:
        raise SuiteError(f"{where} needs `{key}`")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in cast("list[Any]", value)
    ):
        raise SuiteError(f"`{where}` is a list of text")
    return tuple(cast("list[str]", value))


def _mapping(value: Any, where: str) -> dict[str, str]:
    return {str(key): _text(item, f"{where}.{key}") for key, item in _table(value, where).items()}


def _count(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SuiteError(f"`{where}` is a whole number")
    return value


def _seconds(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SuiteError(f"`{where}` is a number of seconds")
    return float(value)
