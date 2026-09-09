"""The packer: what rides along in `dist/void`, and where `rg` comes from."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from scripts.pack import RIPGREP_TARGETS, PackError, fetch_ripgrep, flags, registry_modules


def test_every_agent_the_registry_can_run_is_a_hidden_import() -> None:
    """The registry scans the shelf at run time, so the bundler cannot see
    its modules; each has to be named or `/agent` lists an empty shelf."""
    from cli.registry import BUILTIN, Registry

    modules = registry_modules()
    assert modules == ("cli.agents.dummy_weather", "cli.agents.universal", "cli.agents.weather")
    scanned = {f"cli.agents.{info.id}" for info in Registry(sources=(BUILTIN,)).entries}
    assert scanned == set(modules)  # what the glob names is what the scan finds


def test_the_flags_carry_what_a_bundler_cannot_see() -> None:
    written = flags(None)
    assert written[-1] == "cli/main.py"

    def preceded_by(value: str) -> str:
        return written[written.index(value) - 1]

    # the SDK resolves its transports by name, so its packages go whole
    for package in ("mcp.client", "mcp.server", "mcp.shared"):
        assert preceded_by(package) == "--collect-submodules"
    # mcp.cli drags in typer, which the shell never uses
    assert preceded_by("mcp.cli") == "--exclude-module"
    for module in registry_modules():
        assert module in written


def test_ripgrep_rides_along_with_the_platforms_own_separator() -> None:
    """`--add-binary src<sep>dest`, and the separator is `;` on Windows."""
    carried = flags(Path("/somewhere/rg"))
    assert f"/somewhere/rg{os.pathsep}." in carried
    assert "--add-binary" not in flags(None)


def test_a_platform_with_no_pinned_ripgrep_is_told_so(tmp_path: Path) -> None:
    """Rather than shipping a binary with a silently missing search."""
    original = RIPGREP_TARGETS.copy()
    RIPGREP_TARGETS.clear()
    try:
        with pytest.raises(PackError, match="no pinned ripgrep"):
            fetch_ripgrep(tmp_path)
    finally:
        RIPGREP_TARGETS.update(original)


def test_this_platform_is_one_the_release_matrix_covers() -> None:
    """A guard on the pinned table: the machine running the suite is one of
    the five the matrix builds, so a typo in a key is caught here."""
    import platform

    if sys.platform not in ("darwin", "linux", "win32"):
        pytest.skip("not a platform the matrix builds")
    assert (platform.system(), platform.machine()) in RIPGREP_TARGETS
