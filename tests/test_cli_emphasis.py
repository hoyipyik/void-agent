"""Emphasis in a reply written without spaces: `**氣溫：**18.3°C` is bold,
as its writer meant, and text with no CJK beside the run reads exactly as
CommonMark has it."""

from __future__ import annotations

from pathlib import Path

import pytest
from cli.widgets.emphasis import parser
from markdown_it import MarkdownIt
from tests.test_cli_app import finished, make_app, scripted
from textual.content import Content
from textual.widgets._markdown import MarkdownBlock


def inline(source: str) -> str:
    return parser().renderInline(source)


def test_bold_that_ends_in_cjk_punctuation_closes_before_the_text_after_it() -> None:
    assert inline("**氣溫：**18.3°C") == "<strong>氣溫：</strong>18.3°C"
    assert inline("**注意！**請帶傘") == "<strong>注意！</strong>請帶傘"


def test_bold_that_starts_with_a_cjk_bracket_opens_inside_a_sentence() -> None:
    assert inline("這是**（括號）**文字") == "這是<strong>（括號）</strong>文字"
    assert inline("**「引號」**後面") == "<strong>「引號」</strong>後面"


def test_ascii_punctuation_inside_the_run_holds_beside_a_cjk_character() -> None:
    assert inline("溫度**(°C)**如下") == "溫度<strong>(°C)</strong>如下"
    assert inline("他說**「好」.**然後") == "他說<strong>「好」.</strong>然後"


def test_italic_and_strikethrough_follow_the_same_rule() -> None:
    assert inline("*斜體。*後面") == "<em>斜體。</em>後面"
    assert inline("~~刪除。~~後面") == "<s>刪除。</s>後面"


def test_kana_and_hangul_count_as_cjk() -> None:
    assert inline("**です。**それ") == "<strong>です。</strong>それ"
    assert inline("**날씨:**맑음") == "<strong>날씨:</strong>맑음"


def test_an_underscore_still_cannot_split_a_word() -> None:
    assert inline("這是_強調_文字") == "這是_強調_文字"
    assert inline("snake_case_name") == "snake_case_name"


@pytest.mark.parametrize(
    "source",
    [
        "**Note:**text",  # CommonMark leaves this one raw, and so does this
        "**Note:** text",
        "a * b * c",
        "2*3*4",
        "*(*nested*)*",
        "**bold** and _italic_ and `code`",
        "__init__ and snake_case_name",
        "~~gone~~ and ~not~",
        '*"quoted"*text and text*"quoted"*',
        "***both*** **a*b*c**",
        "| a | b |\n|---|---|\n| **x:**y | *z* |",
        "- **Wind:** 26.6 km/h\n- **Rain:**none",
    ],
)
def test_text_with_no_cjk_beside_the_run_reads_as_commonmark_has_it(source: str) -> None:
    assert parser().render(source) == MarkdownIt("gfm-like").render(source)


async def test_a_reply_renders_bold_that_ends_in_cjk_punctuation(tmp_path: Path) -> None:
    """What a model answering in Chinese writes all day: a list of
    `**label：**value` rows. None of the asterisks reach the screen."""
    app = make_app(tmp_path, lambda _config: scripted("- **氣溫：**18.3°C\n- **體感：**14.5°C"))
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await finished(app)
        await pilot.pause()
        shown = [
            block.content.plain
            for block in app.shell.replies()[0].query(MarkdownBlock)
            if isinstance(block.content, Content) and block.content.plain
        ]
    assert shown == ["氣溫：18.3°C", "體感：14.5°C"]


async def test_a_replayed_reply_renders_the_same_way(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        app.shell.session.append("user", [{"type": "text", "text": "hi"}])
        app.shell.session.append("assistant", [{"type": "text", "text": "**氣溫：**18.3°C"}])
        app.store.save(app.shell.session)
        await app.shell.reopen(app.shell.session.id)
        await pilot.pause()
        shown = [
            block.content.plain
            for block in app.shell.replies()[0].query(MarkdownBlock)
            if isinstance(block.content, Content) and block.content.plain
        ]
    assert shown == ["氣溫：18.3°C"]
