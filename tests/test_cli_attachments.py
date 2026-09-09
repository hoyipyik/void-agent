"""Attachments in the terminal: a dragged path, an @path mention, the OS
clipboard, `/attach` — each a `file` part on the message the model reads
(a text file as text). The composer keeps pasted newlines and takes a
trailing backslash + Enter as a newline."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

from cli.app import VoidApp
from cli.clipboard import Clipboard, Writer
from cli.config import Config
from cli.session import SessionStore
from cli.session.attachments import Attachment, mentions, paths_in, read_attachment
from textual import events

from void_agent import Agent, ImageContent, Message, ScriptedLlm, TextContent, say

CONFIGURED = Config(provider="anthropic", anthropic_api_key="sk-test")
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"


def png(tmp_path: Path, name: str = "shot.png") -> Path:
    path = tmp_path / name
    path.write_bytes(PNG_BYTES)
    return path


# ── reading files ─────────────────────────────────────────────────────────


def test_an_image_file_becomes_a_file_part_with_a_data_url(tmp_path: Path) -> None:
    attachment = read_attachment(png(tmp_path))
    assert isinstance(attachment, Attachment)
    assert attachment.part() == {
        "type": "file",
        "mediaType": "image/png",
        "filename": "shot.png",
        "url": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode(),
    }


def test_pdfs_and_text_files_attach_too(tmp_path: Path) -> None:
    (tmp_path / "r.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "a.py").write_text("print(1)\n")
    (tmp_path / "notes").write_text("plain words\n")
    kinds = [
        getattr(read_attachment(tmp_path / name), "media_type", None)
        for name in ("r.pdf", "a.py", "notes")
    ]
    assert kinds == ["application/pdf", "text/x-python", "text/plain"]


def test_what_cannot_be_attached_says_why(tmp_path: Path) -> None:
    assert "no such file" in str(read_attachment(tmp_path / "missing.png"))
    (tmp_path / "a.zip").write_bytes(b"\x00\x01\x02\xff")
    assert "unsupported" in str(read_attachment(tmp_path / "a.zip"))
    (tmp_path / "large.png").write_bytes(b"\x00" * (5 * 1024 * 1024 + 1))
    assert isinstance(read_attachment(tmp_path / "large.png"), Attachment)  # under 10 MB
    (tmp_path / "big.png").write_bytes(b"\x00" * (10 * 1024 * 1024 + 1))
    assert "too large" in str(read_attachment(tmp_path / "big.png"))
    (tmp_path / "empty.png").write_bytes(b"")
    assert "empty" in str(read_attachment(tmp_path / "empty.png"))


# ── what a paste is ───────────────────────────────────────────────────────


def test_a_paste_of_absolute_paths_is_paths(tmp_path: Path) -> None:
    first = png(tmp_path, "one.png")
    second = png(tmp_path, "two two.png")
    assert paths_in(f"{first} ") == [first]
    assert paths_in(f"'{second}'") == [second]
    assert paths_in(str(second).replace(" ", "\\ ")) == [second]
    assert paths_in(f"{first}\n{second}\n") == [first, second]


def test_a_paste_of_words_is_words(tmp_path: Path) -> None:
    first = png(tmp_path, "one.png")
    assert paths_in("hello world") == []
    assert paths_in(f"{first}\nnot a path") == []
    assert paths_in("one.png") == []  # relative names are words, not files
    assert paths_in("") == []


def test_mentions_attach_files_and_leave_other_ats_alone(tmp_path: Path) -> None:
    shot = png(tmp_path)
    assert mentions(f"look at @{shot} please") == ("look at please", [shot])
    assert mentions("email @bob about it") == ("email @bob about it", [])
    assert mentions(f"@{shot}") == ("", [shot])


# ── the clipboard ─────────────────────────────────────────────────────────


def fake_runner(outputs: dict[str, bytes]) -> Callable[[list[str]], bytes | None]:
    def run(command: list[str]) -> bytes | None:
        return next((out for key, out in outputs.items() if key in " ".join(command)), None)

    return run


def test_the_macos_clipboard_yields_a_copied_file_first(tmp_path: Path) -> None:
    shot = png(tmp_path)
    clipboard = Clipboard(
        runner=fake_runner({"furl": f"{shot}\n".encode(), "PNGf": "«data PNGf00»".encode()}),
        platform="darwin",
    )
    read = clipboard.read()
    assert isinstance(read, list)
    assert [attachment.name for attachment in read] == ["shot.png"]


def test_the_macos_clipboard_yields_a_png_as_hex() -> None:
    clipboard = Clipboard(
        runner=fake_runner(
            {"PNGf": "«data PNGf".encode() + PNG_BYTES.hex().encode() + "»\n".encode()}
        ),
        platform="darwin",
    )
    read = clipboard.read()
    assert isinstance(read, list)
    assert read[0].media_type == "image/png"
    assert read[0].data == PNG_BYTES


def test_an_empty_clipboard_says_so() -> None:
    clipboard = Clipboard(runner=fake_runner({}), platform="darwin")
    assert clipboard.read() == "nothing to paste: no file or image on the clipboard"


def test_the_linux_clipboard_reads_uri_lists_and_pngs(tmp_path: Path) -> None:
    shot = png(tmp_path)
    clipboard = Clipboard(
        runner=fake_runner({"text/uri-list": f"file://{shot}\r\n".encode()}), platform="linux"
    )
    read = clipboard.read()
    assert isinstance(read, list) and read[0].name == "shot.png"
    clipboard = Clipboard(runner=fake_runner({"image/png": PNG_BYTES}), platform="linux")
    read = clipboard.read()
    assert isinstance(read, list) and read[0].data == PNG_BYTES


def fake_writer(taken: list[tuple[str, str]], missing: frozenset[str] = frozenset()) -> Writer:
    """A writer that records what each tool was handed; a tool named in
    `missing` is not installed."""

    def write(command: list[str], data: bytes) -> bool:
        if command[0] in missing:
            return False
        taken.append((command[0], data.decode("utf-8")))
        return True

    return write


def test_copying_writes_the_platforms_own_clipboard_tool() -> None:
    """Textual's copy is an OSC 52 escape, which macOS Terminal ignores and
    iTerm2 refuses by default: the text goes to pbcopy and its kin too."""
    taken: list[tuple[str, str]] = []
    assert Clipboard(writer=fake_writer(taken), platform="darwin").write("héllo")
    assert taken == [("pbcopy", "héllo")]
    taken.clear()
    assert Clipboard(writer=fake_writer(taken), platform="linux").write("one")
    assert taken == [("wl-copy", "one")]
    taken.clear()
    without_wayland = fake_writer(taken, missing=frozenset({"wl-copy"}))
    assert Clipboard(writer=without_wayland, platform="linux").write("two")
    assert taken == [("xclip", "two")]
    taken.clear()
    assert Clipboard(writer=fake_writer(taken), platform="win32").write("three")
    assert taken[0][0] == "powershell" and taken[0][1] == "three"
    nothing = fake_writer(taken, missing=frozenset({"wl-copy", "xclip"}))
    assert not Clipboard(writer=nothing, platform="linux").write("lost")


# ── in the app ────────────────────────────────────────────────────────────


def make_app(
    tmp_path: Path, seen: list[list[Message]], clipboard: Clipboard | None = None
) -> VoidApp:
    def build(_config: Config) -> Agent:
        def remember(history: list[Message]) -> list[Message]:
            seen.append(list(history))
            return list(history)

        return Agent(ScriptedLlm([say("ok")]), "void", "test").prompt(remember)

    return VoidApp(
        build,
        store=SessionStore(tmp_path / "sessions"),
        config=CONFIGURED,
        config_file=tmp_path / "config.json",
        clipboard=clipboard or Clipboard(runner=fake_runner({}), platform="darwin"),
    )


async def finished(app: VoidApp) -> None:
    await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]


def user_parts(app: VoidApp) -> list[dict[str, object]]:
    return app.store.load(app.shell.session.id).messages[0].parts


async def test_a_pasted_path_is_attached_and_sent_with_the_message(tmp_path: Path) -> None:
    shot = png(tmp_path)
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)
    async with app.run_test() as pilot:
        app.shell.composer.post_message(events.Paste(f"{shot} "))
        await pilot.pause()
        assert [attachment.name for attachment in app.shell.pending] == ["shot.png"]
        assert app.shell.composer.text == ""
        await pilot.press(*"what is this", "enter")
        await finished(app)
        await pilot.pause()
        assert app.shell.pending == []
    assert [part["type"] for part in user_parts(app)] == ["file", "text"]
    assert seen[0][0] == Message.user(
        ImageContent(data=PNG_BYTES, media_type="image/png"), "what is this"
    )


async def test_an_at_path_mention_attaches_the_file(tmp_path: Path) -> None:
    shot = png(tmp_path)
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)
    async with app.run_test() as pilot:
        app.shell.composer.text = f"look @{shot} closely"
        await pilot.press("enter")
        await finished(app)
        await pilot.pause()
    parts = user_parts(app)
    assert [part["type"] for part in parts] == ["file", "text"]
    assert parts[1]["text"] == "look closely"


async def test_ctrl_v_pastes_an_image_from_the_clipboard(tmp_path: Path) -> None:
    seen: list[list[Message]] = []
    clipboard = Clipboard(
        runner=fake_runner(
            {"PNGf": b"\xc2\xabdata PNGf" + PNG_BYTES.hex().encode() + b"\xc2\xbb"}
        ),
        platform="darwin",
    )
    app = make_app(tmp_path, seen, clipboard)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+v")
        await finished(app)
        await pilot.pause()
        assert [attachment.media_type for attachment in app.shell.pending] == ["image/png"]
        await pilot.press(*"see", "enter")
        await finished(app)
        await pilot.pause()
    assert [part["type"] for part in user_parts(app)] == ["file", "text"]


async def test_slash_attach_and_slash_detach(tmp_path: Path) -> None:
    shot = png(tmp_path)
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)
    async with app.run_test() as pilot:
        app.shell.composer.text = f"/attach {shot}"
        await pilot.press("enter")
        await pilot.pause()
        assert [attachment.name for attachment in app.shell.pending] == ["shot.png"]
        await pilot.press(*"/detach", "enter")
        await pilot.pause()
        assert app.shell.pending == []
        app.shell.composer.text = f"/attach {tmp_path / 'missing.png'}"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query(".error")


async def test_slash_paste_reads_the_clipboard(tmp_path: Path) -> None:
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)  # an empty clipboard
    async with app.run_test() as pilot:
        await pilot.press(*"/paste", "enter")
        await finished(app)
        await pilot.pause()
        assert app.shell.pending == []
        assert app.query(".error")


async def test_a_text_file_arrives_as_text_the_model_reads(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)\n")
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)
    async with app.run_test() as pilot:
        app.shell.composer.text = f"review @{tmp_path / 'a.py'}"
        await pilot.press("enter")
        await finished(app)
        await pilot.pause()
    assert seen[0][0] == Message.user(TextContent("[file: a.py]\n```\nprint(1)\n```\nreview"))


async def test_pasted_text_keeps_its_lines_and_backslash_enter_adds_one(tmp_path: Path) -> None:
    seen: list[list[Message]] = []
    app = make_app(tmp_path, seen)
    async with app.run_test() as pilot:
        app.shell.composer.post_message(events.Paste("line one\nline two"))
        await pilot.pause()
        assert app.shell.composer.text == "line one\nline two"
        app.shell.composer.text = "first\\"
        await pilot.press("enter")
        await pilot.pause()
        assert app.shell.composer.text == "first\n"
        assert seen == []
        await pilot.press(*"second", "enter")
        await finished(app)
        await pilot.pause()
    assert seen[0][0] == Message.user("first\nsecond")
