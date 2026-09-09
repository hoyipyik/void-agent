"""The protocol rendered in the terminal: tool chips, the plan and
reflection cards, a tool's own progress, the question card answered in
place — a gate's signature on the card, the model's question in the
composer — and Stop."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from cli.app import VoidApp
from cli.config import Config
from cli.session import SessionStore
from cli.shell import BUSY_PROMPT, PROMPT
from cli.widgets import AskCard, Composer, DataCard, PlanCard, ReflectionCard, ToolChip
from pydantic import BaseModel, ConfigDict
from textual.pilot import Pilot
from textual.widgets import OptionList, Static

from void_agent import (
    HUMAN,
    Agent,
    EventSender,
    ModelStep,
    Rejected,
    ScriptedLlm,
    ScriptedStep,
    Tool,
    Usage,
    call,
    say,
    tool,
    tool_call,
)

CONFIGURED = Config(provider="anthropic", anthropic_api_key="sk-test")


class CityQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


@tool(description="weather")
async def weather(input: CityQuery) -> dict[str, Any]:
    if input.city == "nowhere":
        raise Rejected("unknown city: nowhere")
    return {"sky": "clear"}


@tool(description="a workflow that reports progress")
async def gather(input: CityQuery, events: EventSender) -> dict[str, Any]:
    await events.progress("gather", {"city": input.city, "stock": 60})
    return {"ok": True}


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total: float


def needs_signature(draft: Draft) -> str | None:
    return f"a draft of {draft.total:.2f} needs a signature" if draft.total > 100 else None


def gated_draft(ran: list[float]) -> Tool:
    @tool(description="create a draft", approval=needs_signature)
    async def create_draft(input: Draft) -> dict[str, str]:
        ran.append(input.total)
        return {"draft_id": "D-1"}

    return create_draft


class Nothing(BaseModel):
    model_config = ConfigDict(extra="forbid")


@tool(description="never returns")
async def forever(input: Nothing) -> dict[str, str]:
    await asyncio.sleep(3600)
    return {}


def app_with(
    tmp_path: Path,
    script: list[ScriptedStep],
    *tools: Tool,
    plan: bool = False,
    reflection: bool = False,
) -> VoidApp:
    def build(_config: Config) -> Agent:
        agent = Agent(ScriptedLlm(list(script)), "void", "test")
        agent = agent.prompt(lambda history: list(history)).tool(HUMAN)
        if plan:
            agent = agent.with_plan()
        if reflection:
            agent = agent.with_reflection()
        for capability in tools:
            agent = agent.tool(capability)
        return agent

    return VoidApp(
        build,
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
    )


async def finished(app: VoidApp) -> None:
    await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
    await asyncio.sleep(0)


async def until(pilot: Pilot[None], condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition never held")
        await pilot.pause(0.02)


def stored_parts(app: VoidApp) -> list[dict[str, Any]]:
    return app.store.load(app.shell.session.id).messages[1].parts


def card_open(app: VoidApp) -> bool:
    """A question card is up and its options hold the keys."""
    return bool(app.query(AskCard)) and isinstance(app.focused, OptionList)


async def test_a_tool_call_renders_as_a_chip_that_finishes_with_its_output(tmp_path: Path) -> None:
    app = app_with(tmp_path, [call("weather", {"city": "tokyo"}), say("clear skies")], weather)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        chips = list(app.query(ToolChip))
        assert [(chip.tool_name, chip.state) for chip in chips] == [
            ("weather", "output-available")
        ]
        assert [reply.source for reply in app.shell.replies()] == ["clear skies"]
    assert [part["type"] for part in stored_parts(app)] == ["dynamic-tool", "text"]


class Note(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


@tool(description="echoes")
async def echo(input: Note) -> dict[str, Any]:
    return {"echo": input.text, "items": ["[a]", "$b", "[/]"]}


async def test_brackets_in_tool_data_and_questions_render_verbatim(tmp_path: Path) -> None:
    """A tool's data and the model's words are not markup: `[`, `]` and `$`
    reach the screen as they are, and never break the turn."""
    text = 'note: [x] "$y" [/] [b]bold[/b]'
    script: list[ScriptedStep] = [
        call("echo", {"text": text}),
        ModelStep(
            text="",
            tool_calls=(tool_call("ask_user", {"kind": "input", "question": "Which [one]?"}),),
        ),
        say("done [ok]"),
    ]
    app = app_with(tmp_path, script, echo)
    async with app.run_test() as pilot:
        await pilot.press(*"go", "enter")
        await until(pilot, lambda: bool(app.query(AskCard)))
        card = app.query_one(AskCard)
        assert card.question == "Which [one]?"
        assert card.query_one(".ask-question", Static).content == "Which [one]?"
        composer = app.query_one("#composer", Composer)
        await until(pilot, lambda: not composer.disabled)
        await pilot.press(*"that", "enter")
        await finished(app)
        await pilot.pause()
        chip = app.query_one(ToolChip)
        body = str(chip.query_one(".fold-body", Static).content)
        assert "[x]" in body and "[b]bold[/b]" in body and "[/]" in body
        assert '"[a]"' in body and '"$b"' in body
        assert [reply.source for reply in app.shell.replies()] == ["done [ok]"]
    assert stored_parts(app)[0]["output"]["echo"] == text


async def test_a_tool_error_renders_as_a_failed_chip(tmp_path: Path) -> None:
    app = app_with(tmp_path, [call("weather", {"city": "nowhere"}), say("no such place")], weather)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        chip = app.query_one(ToolChip)
        assert chip.state == "output-error"
        assert not chip.collapsed  # a failure opens itself, as the web's chip does


async def test_every_plan_update_shows_the_plan_where_it_happened(tmp_path: Path) -> None:
    first = [{"id": "1", "title": "look up", "status": "in_progress"}]
    second = [{"id": "1", "title": "look up", "status": "completed", "note": "done"}]
    script: list[ScriptedStep] = [
        call("update_plan", {"items": first}),
        call("update_plan", {"items": second}),
        say("all done"),
    ]
    app = app_with(tmp_path, script, plan=True)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        cards = list(app.query(PlanCard))
        # one card per update, each as the plan stood then; the last is current
        assert [[item["status"] for item in card.items] for card in cards] == [
            ["in_progress"],
            ["completed"],
        ]
        assert cards[-1].items[0]["note"] == "done"


async def test_a_reflection_renders_as_a_card(tmp_path: Path) -> None:
    script: list[ScriptedStep] = [
        call(
            "reflect",
            {
                "facts": "only 60 in stock",
                "verdict": "adjust",
                "problems": ["short by 40"],
                "adjustment": "draft 60",
            },
        ),
        say("adjusted"),
    ]
    app = app_with(tmp_path, script, reflection=True)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        card = app.query_one(ReflectionCard)
        assert card.verdict == "adjust"


async def test_a_tools_own_progress_renders_as_a_data_card(tmp_path: Path) -> None:
    app = app_with(tmp_path, [call("gather", {"city": "tokyo"}), say("gathered")], gather)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        card = app.query_one(DataCard)
        assert card.kind == "gather"


async def test_a_gated_call_is_signed_on_the_card_and_runs_in_place(tmp_path: Path) -> None:
    ran: list[float] = []
    script: list[ScriptedStep] = [call("create_draft", {"total": 270}), say("D-1 is ready")]
    app = app_with(tmp_path, script, gated_draft(ran))
    async with app.run_test() as pilot:
        await pilot.press(*"draft it", "enter")
        await until(pilot, lambda: card_open(app))
        card = app.query_one(AskCard)
        assert card.signature
        assert ran == []  # held until signed
        await pilot.press("enter")  # the first row: yes, run it
        await finished(app)
        await pilot.pause()
        assert ran == [270.0]
        assert card.answered is True
        assert [reply.source for reply in app.shell.replies()] == ["D-1 is ready"]
    parts = stored_parts(app)
    assert [part["type"] for part in parts] == ["dynamic-tool", "data-ask", "data-answer", "text"]
    assert parts[0]["output"] == {"draft_id": "D-1"}
    assert parts[2]["data"]["value"] is True


async def test_a_declined_call_never_runs(tmp_path: Path) -> None:
    ran: list[float] = []
    script: list[ScriptedStep] = [call("create_draft", {"total": 270}), say("not created")]
    app = app_with(tmp_path, script, gated_draft(ran))
    async with app.run_test() as pilot:
        await pilot.press(*"draft it", "enter")
        await until(pilot, lambda: card_open(app))
        await pilot.press("2")  # the second row: no
        await finished(app)
        await pilot.pause()
        assert ran == []
        assert app.query_one(AskCard).answered is False
    parts = stored_parts(app)
    assert parts[0]["state"] == "output-error"


async def test_the_models_question_is_answered_in_the_composer(tmp_path: Path) -> None:
    script: list[ScriptedStep] = [
        ModelStep(
            text="",
            tool_calls=(tool_call("ask_user", {"kind": "input", "question": "Which colour?"}),),
        ),
        say("blue it is"),
    ]
    app = app_with(tmp_path, script)
    async with app.run_test() as pilot:
        await pilot.press(*"pick", "enter")
        await until(pilot, lambda: bool(app.query(AskCard)))
        assert not isinstance(app.focused, OptionList)  # words are typed, not chosen
        composer = app.query_one("#composer", Composer)
        await until(pilot, lambda: not composer.disabled)
        await pilot.press(*"blue", "enter")
        await finished(app)
        await pilot.pause()
        assert app.query_one(AskCard).answered == "blue"
        assert [reply.source for reply in app.shell.replies()] == ["blue it is"]
    types = [part["type"] for part in stored_parts(app)]
    assert "data-ask" in types and "data-answer" in types
    assert len(app.store.load(app.shell.session.id).messages) == 2  # the answer is not a message


async def test_a_choice_question_offers_its_options_as_buttons(tmp_path: Path) -> None:
    script: list[ScriptedStep] = [
        ModelStep(
            text="",
            tool_calls=(
                tool_call(
                    "ask_user",
                    {"kind": "choice", "question": "Which?", "options": ["red", "blue"]},
                ),
            ),
        ),
        say("blue it is"),
    ]
    app = app_with(tmp_path, script)
    async with app.run_test() as pilot:
        await pilot.press(*"pick", "enter")
        await until(pilot, lambda: card_open(app))
        card = app.query_one(AskCard)
        await pilot.press("down", "enter")  # red, then blue
        await finished(app)
        await pilot.pause()
        assert card.answered == "blue"


async def test_a_choice_card_holds_the_keys_and_other_opens_the_composer(tmp_path: Path) -> None:
    script: list[ScriptedStep] = [
        ModelStep(
            text="",
            tool_calls=(
                tool_call(
                    "ask_user",
                    {"kind": "choice", "question": "Which?", "options": ["red", "blue"]},
                ),
            ),
        ),
        say("green it is"),
    ]
    app = app_with(tmp_path, script)
    async with app.run_test() as pilot:
        await pilot.press(*"pick", "enter")
        await until(pilot, lambda: card_open(app))
        composer = app.query_one("#composer", Composer)
        assert composer.disabled  # the card has the keys; nothing steals them
        await pilot.press("3")  # red, blue, Other
        await until(pilot, lambda: not composer.disabled)
        await pilot.press(*"green", "enter")
        await finished(app)
        await pilot.pause()
        assert app.query_one(AskCard).answered == "green"
        assert [reply.source for reply in app.shell.replies()] == ["green it is"]


async def test_while_a_turn_runs_the_composer_says_it_is_waiting(tmp_path: Path) -> None:
    """The box is closed while the model works — and says so, rather than
    still inviting a question it cannot take."""
    app = app_with(tmp_path, [call("forever", {}), say("never")], forever)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", Composer)
        assert composer.placeholder == PROMPT
        await pilot.press(*"go", "enter")
        await until(pilot, lambda: bool(app.query(ToolChip)))
        assert composer.disabled
        assert composer.placeholder == BUSY_PROMPT
        await pilot.press("escape")
        await finished(app)
        await pilot.pause()
        assert not composer.disabled
        assert composer.placeholder == PROMPT


async def test_escape_stops_the_turn_and_marks_it_cancelled(tmp_path: Path) -> None:
    app = app_with(tmp_path, [call("forever", {}), say("never")], forever)
    async with app.run_test() as pilot:
        await pilot.press(*"go", "enter")
        await until(pilot, lambda: bool(app.query(ToolChip)))
        await pilot.press("escape")
        await finished(app)
        await pilot.pause()
        assert not app.query_one("#composer", Composer).disabled
    parts = stored_parts(app)
    assert parts[-1] == {"type": "data-cancelled", "data": {}}


async def test_a_reopened_session_replays_its_cards_from_parts(tmp_path: Path) -> None:
    ran: list[float] = []
    script: list[ScriptedStep] = [call("create_draft", {"total": 270}), say("D-1 is ready")]
    app = app_with(tmp_path, script, gated_draft(ran))
    async with app.run_test() as pilot:
        await pilot.press(*"draft it", "enter")
        await until(pilot, lambda: card_open(app))
        await pilot.press("enter")
        await finished(app)
        session_id = app.shell.session.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert not app.query(AskCard)
        await app.shell.reopen(session_id)
        await pilot.pause()
        assert [chip.state for chip in app.query(ToolChip)] == ["output-available"]
        assert app.query_one(AskCard).answered is True
        assert [reply.source for reply in app.shell.replies()] == ["D-1 is ready"]


def usage_lines(app: VoidApp) -> list[str]:
    return [str(line.render()) for line in app.query(".usage")]


async def test_each_round_trips_cost_shows_under_it_and_the_bar_says_the_context(
    tmp_path: Path,
) -> None:
    script: list[ScriptedStep] = [
        replace(call("weather", {"city": "tokyo"}), usage=Usage(input=1200, output=45)),
        replace(say("clear skies"), usage=Usage(input=1400, output=12, cache_read=1200)),
    ]
    app = app_with(tmp_path, script, weather)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        assert usage_lines(app) == ["∑ 1.2k in · 45 out", "∑ 1.4k in · 12 out · 1.2k cached"]
        assert app.shell.status.label.endswith(" · 1.4k ctx")
        # Where the round-trip ended: after the text it streamed, before
        # the calls it made.
        assert [part["type"] for part in stored_parts(app)] == [
            "data-usage",
            "dynamic-tool",
            "text",
            "data-usage",
        ]


async def test_a_reopened_session_shows_its_cost_lines_and_context_again(tmp_path: Path) -> None:
    script: list[ScriptedStep] = [replace(say("ok"), usage=Usage(input=800, output=3))]
    app = app_with(tmp_path, script)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        session_id = app.shell.session.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert usage_lines(app) == []
        assert " ctx" not in app.shell.status.label
        await app.shell.reopen(session_id)
        await pilot.pause()
        assert usage_lines(app) == ["∑ 800 in · 3 out"]
        assert app.shell.status.label.endswith(" · 800 ctx")


async def test_a_model_that_reports_nothing_leaves_no_cost_line(tmp_path: Path) -> None:
    app = app_with(tmp_path, [say("ok")])
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        assert usage_lines(app) == []
        assert " ctx" not in app.shell.status.label
