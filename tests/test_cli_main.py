"""The door: what `void` refuses before the shell starts."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest


def entry(monkeypatch: pytest.MonkeyPatch, home: Path) -> Callable[[Sequence[str] | None], None]:
    """`cli.main` reads VOID_HOME at import, so it is reloaded on the override."""
    monkeypatch.setenv("VOID_HOME", str(home))
    import cli.main

    importlib.reload(cli.main)
    return cli.main.main


def test_a_workspace_that_is_not_a_folder_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main = entry(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        main(["--workspace", str(tmp_path / "nope")])
    assert "not a folder" in capsys.readouterr().err


def test_an_agent_that_is_not_there_is_refused_at_the_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main = entry(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        main(["--agent", "nope"])
    assert "no agent named `nope`" in capsys.readouterr().err
