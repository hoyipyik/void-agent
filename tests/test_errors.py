"""Error privacy: what leaves the process goes through `public_text`."""

from __future__ import annotations

from void_agent import INTERNAL_PUBLIC_TEXT, Exhausted, Internal, Rejected, public_text


def test_internal_detail_never_reaches_public_text() -> None:
    error = Internal("call model", RuntimeError("db password=hunter2"))
    assert "hunter2" in str(error)  # logs see the cause
    assert public_text(error) == INTERNAL_PUBLIC_TEXT


def test_readable_errors_speak_for_themselves() -> None:
    assert public_text(Rejected("out of stock")) == "out of stock"
    assert public_text(Exhausted("step limit")) == "step limit"
