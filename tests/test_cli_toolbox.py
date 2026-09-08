"""void's own files server: it answers about its root and nowhere else."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from cli.toolbox import search
from cli.toolbox.files import edit_text, list_dir, move, read_many, read_text, tree_of, write_text
from cli.toolbox.process import run_in
from cli.toolbox.root import MAX_CHARS, ToolboxError, resolve
from cli.toolbox.search import count_in, files_in, search_in

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")


def repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "loop.py").write_text("def step():\n    return 'a needle here'\n")
    (root / "src" / "notes.md").write_text("a needle in the notes\n")
    (root / "README.md").write_text("nothing to see\n")
    return root


@needs_rg
async def test_a_search_answers_with_paths_relative_to_the_root(tmp_path: Path) -> None:
    found = await search_in(repo(tmp_path), "needle")
    assert "src/loop.py:2:" in found
    assert "src/notes.md:1:" in found
    assert str(tmp_path) not in found  # the root is not the model's business


@needs_rg
async def test_a_glob_narrows_the_search(tmp_path: Path) -> None:
    found = await search_in(repo(tmp_path), "needle", glob="*.py")
    assert "loop.py" in found and "notes.md" not in found


@needs_rg
async def test_no_matches_is_an_answer_not_a_failure(tmp_path: Path) -> None:
    assert "no matches" in await search_in(repo(tmp_path), "haystack")


@needs_rg
async def test_a_path_outside_the_root_is_refused(tmp_path: Path) -> None:
    root = repo(tmp_path / "inside")
    (tmp_path / "secret.txt").write_text("a needle nobody asked for\n")
    for attempt in ("..", "../secret.txt", "/etc", str(tmp_path)):
        with pytest.raises(ToolboxError, match="outside"):
            await search_in(root, "needle", path=attempt)


@needs_rg
async def test_the_root_itself_is_allowed(tmp_path: Path) -> None:
    root = repo(tmp_path)
    assert resolve(root, None) == root
    assert resolve(root, "src") == root / "src"
    assert "loop.py" in await search_in(root, "needle", path="src")


@needs_rg
async def test_a_pattern_that_looks_like_a_flag_stays_a_pattern(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "dashes.txt").write_text("--version is written here\n")
    found = await search_in(root, "--version", fixed=True)
    assert "dashes.txt" in found


@needs_rg
async def test_a_long_line_is_cut_rather_than_spent(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "long.txt").write_text("needle " + "x" * 5000 + "\n")
    found = await search_in(root, "needle", glob="long.txt")
    assert "chars)" in found and len(found) < 1000


@needs_rg
async def test_counting_is_the_cheap_way_to_size_a_search(tmp_path: Path) -> None:
    counted = await count_in(repo(tmp_path), "needle")
    assert "src/loop.py:1" in counted


@needs_rg
async def test_listing_files_honours_the_root_and_the_glob(tmp_path: Path) -> None:
    listed = await files_in(repo(tmp_path), glob="*.md")
    assert "src/notes.md" in listed and "loop.py" not in listed


def test_reading_a_file_stays_inside_the_root(tmp_path: Path) -> None:
    root = repo(tmp_path / "inside")
    (tmp_path / "secret.txt").write_text("not yours\n")
    assert "needle" in read_text(root, "src/loop.py")
    for attempt in ("../secret.txt", "/etc/hosts", ".."):
        with pytest.raises(ToolboxError, match=r"outside|directory"):
            read_text(root, attempt)


def test_head_and_tail_read_less_than_the_whole_file(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "many.txt").write_text("\n".join(str(n) for n in range(100)))
    assert read_text(root, "many.txt", head=3) == "0\n1\n2"
    assert read_text(root, "many.txt", tail=2) == "98\n99"


def test_a_directory_listing_marks_its_directories(tmp_path: Path) -> None:
    listed = list_dir(repo(tmp_path))
    assert "src/" in listed
    assert "README.md" in listed and "bytes" in listed


def test_writing_stays_inside_the_root_and_makes_its_parents(tmp_path: Path) -> None:
    root = repo(tmp_path)
    said = write_text(root, "drafts/new.md", "hello\n")
    assert "drafts/new.md" in said
    assert (root / "drafts" / "new.md").read_text() == "hello\n"
    with pytest.raises(ToolboxError, match="outside"):
        write_text(root, "../escaped.txt", "no")
    assert not (root.parent / "escaped.txt").exists()


def test_reading_a_directory_says_to_list_it_instead(tmp_path: Path) -> None:
    with pytest.raises(ToolboxError, match="list_directory"):
        read_text(repo(tmp_path), "src")


# ── the rest of what a person expects a file server to do ───────────────


def test_an_edit_replaces_exactly_what_it_was_given(tmp_path: Path) -> None:
    root = repo(tmp_path)
    said = edit_text(root, "src/loop.py", "'a needle here'", "'a haystack'")
    assert "src/loop.py" in said
    assert (root / "src" / "loop.py").read_text() == "def step():\n    return 'a haystack'\n"


def test_an_edit_that_matches_nothing_says_so_and_changes_nothing(tmp_path: Path) -> None:
    root = repo(tmp_path)
    before = (root / "src" / "loop.py").read_text()
    with pytest.raises(ToolboxError, match="does not appear"):
        edit_text(root, "src/loop.py", "not in the file", "x")
    assert (root / "src" / "loop.py").read_text() == before


def test_an_ambiguous_edit_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "twice.txt").write_text("same\nsame\n")
    with pytest.raises(ToolboxError, match="appears 2 times"):
        edit_text(root, "twice.txt", "same", "other")
    assert (root / "twice.txt").read_text() == "same\nsame\n"
    said = edit_text(root, "twice.txt", "same", "other", replace_all=True)
    assert "2" in said
    assert (root / "twice.txt").read_text() == "other\nother\n"


def test_an_edit_stays_inside_the_root(tmp_path: Path) -> None:
    root = repo(tmp_path / "inside")
    (tmp_path / "outside.txt").write_text("secret\n")
    with pytest.raises(ToolboxError, match="outside"):
        edit_text(root, "../outside.txt", "secret", "changed")
    assert (tmp_path / "outside.txt").read_text() == "secret\n"


def test_several_files_come_back_in_one_answer(tmp_path: Path) -> None:
    read = read_many(repo(tmp_path), ["src/loop.py", "README.md", "gone.txt"])
    assert "src/loop.py" in read and "needle" in read
    assert "README.md" in read
    assert "gone.txt" in read and "does not exist" in read  # one failure is not the lot


def test_a_tree_shows_the_shape_without_the_contents(tmp_path: Path) -> None:
    tree = tree_of(repo(tmp_path))
    assert "src/" in tree and "loop.py" in tree
    assert "def step" not in tree


def test_a_tree_stops_at_the_depth_it_was_given(tmp_path: Path) -> None:
    root = repo(tmp_path)
    (root / "src" / "deep").mkdir()
    (root / "src" / "deep" / "buried.py").write_text("x\n")
    assert "buried.py" not in tree_of(root, depth=1)
    assert "buried.py" in tree_of(root, depth=3)


def test_moving_a_file_keeps_both_ends_inside_the_root(tmp_path: Path) -> None:
    root = repo(tmp_path)
    said = move(root, "README.md", "docs/README.md")
    assert "docs/README.md" in said
    assert (root / "docs" / "README.md").exists() and not (root / "README.md").exists()
    for source, destination in (("src/loop.py", "../escaped.py"), ("../x", "src/y")):
        with pytest.raises(ToolboxError, match=r"outside|does not exist"):
            move(root, source, destination)


def test_moving_onto_an_existing_file_is_refused(tmp_path: Path) -> None:
    root = repo(tmp_path)
    with pytest.raises(ToolboxError, match="already"):
        move(root, "README.md", "src/notes.md")


# ── running something ───────────────────────────────────────────────────


async def test_a_command_runs_in_the_root_and_says_how_it_went(tmp_path: Path) -> None:
    said = await run_in(repo(tmp_path), sys.executable, ["-c", "print('hello')"])
    assert "hello" in said and "exit 0" in said


async def test_a_failing_command_reports_its_code_and_its_words(tmp_path: Path) -> None:
    said = await run_in(
        repo(tmp_path),
        sys.executable,
        ["-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(3)"],
    )
    assert "exit 3" in said and "bad" in said  # stderr is part of the answer


async def test_there_is_no_shell_so_a_redirect_is_not_a_redirect(tmp_path: Path) -> None:
    """The arguments are a list, never a string a shell expands: what the
    person signs is what runs, with nothing hidden in the middle."""
    root = repo(tmp_path)
    said = await run_in(root, sys.executable, ["-c", "import sys; print(sys.argv[1])", "a > b"])
    assert "a > b" in said  # passed through, not interpreted
    assert not (root / "b").exists()


async def test_a_command_that_is_not_there_is_a_readable_refusal(tmp_path: Path) -> None:
    with pytest.raises(ToolboxError, match="not found"):
        await run_in(repo(tmp_path), "definitely-not-a-command-here", [])


async def test_the_working_directory_stays_inside_the_root(tmp_path: Path) -> None:
    root = repo(tmp_path)
    where = await run_in(root, sys.executable, ["-c", "import os; print(os.getcwd())"], cwd="src")
    assert "src" in where
    with pytest.raises(ToolboxError, match="outside"):
        await run_in(root, sys.executable, ["-c", "print(1)"], cwd="..")


async def test_a_command_that_will_not_stop_is_stopped(tmp_path: Path) -> None:
    with pytest.raises(ToolboxError, match="longer than"):
        await run_in(
            repo(tmp_path), sys.executable, ["-c", "import time; time.sleep(30)"], timeout=0.5
        )


async def test_a_torrent_of_output_is_cut(tmp_path: Path) -> None:
    said = await run_in(repo(tmp_path), sys.executable, ["-c", "print('x' * 200000)"])
    assert len(said) < MAX_CHARS + 500


def test_the_packed_binary_finds_the_ripgrep_it_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen, `rg` sits beside the bundle rather than on the PATH."""
    carried = tmp_path / "rg"
    carried.write_text("#!/bin/sh\nexit 0\n")
    carried.chmod(0o755)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert search.ripgrep() == str(carried)

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert search.ripgrep() == shutil.which("rg")
