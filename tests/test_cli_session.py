"""The CLI's sessions: parts on disk, the model-facing projection in memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cli.session import CONTEXT_MESSAGES, SessionStore

from void_agent import ImageContent, Message

PNG = {"type": "file", "mediaType": "image/png", "url": "data:image/png;base64,aW1hZ2UtdGVzdA=="}


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


def test_a_session_round_trips_through_the_store(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.new()
    session.append("user", [text("hi there")])
    session.append("assistant", [text("hello")])
    store.save(session)
    loaded = store.load(session.id)
    assert loaded == session
    assert loaded.title == "hi there"
    assert [s.id for s in store.list()] == [session.id]


def test_truncating_drops_a_message_and_everything_after_it(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    session.append("user", [text("one")])
    session.append("assistant", [text("ok")])
    session.append("user", [text("two")])
    session.append("assistant", [text("ok")])
    assert session.user_indexes() == [0, 2]
    session.truncate(2)
    assert session.history() == [Message.user("one"), Message.assistant("ok")]
    session.truncate(0)
    assert session.messages == []
    assert session.title == "New conversation"


def test_history_is_the_model_facing_projection_with_attachments_intact(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    session.append("user", [text("look"), PNG])
    session.append("assistant", [text("a square")])
    assert session.history() == [
        Message.user("look", ImageContent(data=b"image-test", media_type="image/png")),
        Message.assistant("a square"),
    ]


def test_history_is_the_tail_of_the_context_window(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    for index in range(CONTEXT_MESSAGES + 5):
        session.append("user" if index % 2 == 0 else "assistant", [text(f"m{index}")])
    history = session.history()
    assert len(history) == CONTEXT_MESSAGES
    assert history[-1] == Message.user(f"m{CONTEXT_MESSAGES + 4}")


def test_history_caps_one_tool_result_at_the_limit_it_is_given(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    session.append("user", [text("search")])
    huge = "x" * 10_000
    result: dict[str, Any] = {
        "type": "dynamic-tool",
        "toolCallId": "c",
        "toolName": "search",
        "state": "output-available",
        "input": {},
        "output": huge,
    }
    session.append("assistant", [result])
    assert session.history()[1] == Message.assistant(f"[tool search({{}}) → {json.dumps(huge)}]")
    capped = session.history(tool_output_limit=40)
    assert capped[1] == Message.assistant(f"[tool search({{}}) → {json.dumps(huge)[:40]}…]")


def test_a_message_with_nothing_to_say_is_left_out_of_history(tmp_path: Path) -> None:
    session = SessionStore(tmp_path).new()
    session.append("user", [text("hi")])
    session.append("assistant", [{"type": "data-step", "data": {"step": 1}}])
    assert session.history() == [Message.user("hi")]


def test_an_emptied_session_leaves_no_file_behind(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = store.new()
    session.append("user", [text("hi")])
    store.save(session)
    assert store.list()
    session.clear()
    store.save(session)
    assert store.list() == []
    assert not (tmp_path / f"{session.id}.json").exists()


def test_sessions_list_newest_first(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    older = store.new()
    older.append("user", [text("first")])
    store.save(older)
    newer = store.new()
    newer.append("user", [text("second")])
    store.save(newer)
    older.append("assistant", [text("reply")])
    store.save(older)  # touched again: rises to the top
    assert [s.title for s in store.list()] == ["first", "second"]


def test_a_new_session_is_not_listed_until_it_says_something(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.new()
    assert store.list() == []
