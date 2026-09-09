"""The account, as the CLI shows it: a token count as people read one,
a round-trip's cost in a line, the session's tally, and the `/status`
line that sums it."""

from __future__ import annotations

from pathlib import Path

from cli.labels import bar_label, duration, tokens, usage_label
from cli.session import SessionStore, Tally
from cli.widgets.panels import tokens_line
from tests.test_cli_app import CONFIGURED

from void_agent import Message, Usage


def usage_part(input: int, output: int, cached: int = 0) -> dict[str, object]:
    return {
        "type": "data-usage",
        "data": {"input": input, "output": output, "cacheRead": cached, "cacheWrite": 0},
    }


def test_tokens_read_the_way_people_say_them() -> None:
    assert [tokens(n) for n in (0, 340, 999, 1_000, 1_234, 9_960, 12_345, 1_234_567)] == [
        "0",
        "340",
        "999",
        "1k",
        "1.2k",
        "10k",
        "12k",
        "1.2M",
    ]


def test_a_round_trips_cost_names_the_cache_only_when_it_served() -> None:
    assert usage_label(Usage(input=1200, output=45)) == "1.2k in · 45 out"
    assert usage_label(Usage(input=1200, output=45, cache_read=900)) == (
        "1.2k in · 45 out · 900 cached"
    )


def test_the_bar_names_the_context_once_a_round_trip_said() -> None:
    assert bar_label(CONFIGURED, "universal") == "universal · Anthropic · claude-opus-5"
    assert bar_label(CONFIGURED, "universal", 12_345) == (
        "universal · Anthropic · claude-opus-5 · 12k ctx"
    )
    assert bar_label(CONFIGURED, "universal", 12_345, 51_234) == (
        "universal · Anthropic · claude-opus-5 · 12k ctx · 51k consumed"
    )


def test_the_session_tallies_every_round_trip_it_kept(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    assert session.tally() == Tally()
    session.append("user", [{"type": "text", "text": "hi"}])
    session.append("assistant", [usage_part(100, 5), {"type": "text", "text": "a"}])
    session.append("user", [{"type": "text", "text": "more"}])
    session.append("assistant", [usage_part(130, 7, cached=100), usage_part(160, 9)])
    assert session.tally() == Tally(
        total=Usage(input=390, output=21, cache_read=100), steps=3, context=160
    )
    assert session.tally().consumed == 411
    session.truncate(2)
    assert session.tally() == Tally(total=Usage(input=100, output=5), steps=1, context=100)


def test_the_status_line_sums_the_session_or_says_nothing_was_counted() -> None:
    assert tokens_line(Tally()).plain == "  tokens    nothing counted yet"
    tally = Tally(
        total=Usage(input=12_345, output=1_234, cache_read=9_000), steps=3, context=4_321
    )
    assert tokens_line(tally).plain == (
        "  tokens    12k in · 1.2k out · 9k cached · 3 steps · context 4.3k"
    )


def test_the_next_turns_model_reads_the_session_without_its_account(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    session.append("user", [{"type": "text", "text": "hi"}])
    session.append("assistant", [usage_part(1200, 45, cached=900), {"type": "text", "text": "a"}])
    session.append("assistant", [usage_part(1400, 12)])
    assert session.history() == [Message.user("hi"), Message.assistant("a")]


def test_a_span_reads_the_way_people_say_it() -> None:
    assert [duration(s) for s in (0, 0.4, 4, 12.6, 60, 72, 3600, 3720, 7199)] == [
        "0s",
        "0s",
        "4s",
        "13s",
        "1m",
        "1m 12s",
        "1h",
        "1h 2m",
        "1h 59m",
    ]
