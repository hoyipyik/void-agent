#!/usr/bin/env python
"""Pack the terminal UI into one binary: `dist/void`.

One definition of the flags, because two would drift: `make cli-build` and
the release matrix (`.github/workflows/ci.yml`) both come here. The
Makefile's shell could not do it alone — `.venv/bin/rg` is
`.venv/Scripts/rg.exe` on Windows, and which modules the agent registry
imports at run time is Python's question to answer, not `sh`'s.

What rides along in the bundle: textual whole; the MCP SDK's client,
server and shared packages whole, since it resolves its transports by
name and a bundler cannot see that (`mcp.cli` is excluded — it drags in
typer, which the shell never uses); every module named in the registry's
catalogue, imported at run time and so invisible too; and `rg`, because
the toolbox searches with it and a binary that has to find one on the
person's PATH is not one binary.

Where `rg` comes from: `--fetch-ripgrep` downloads the official release
for this platform and checks it against the digests pinned below — that
is what the release matrix uses, because the PyPI `ripgrep` wheel exists
for two platforms only and its sdist is ripgrep's Rust source, which
would mean a compiler on every runner. Without the flag the local `.venv`
answers, then PATH; with neither, the binary still reads and lists, and
search says why it cannot.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = "cli/main.py"
NAME = "void"

RIPGREP_VERSION = "15.2.0"
RIPGREP_URL = "https://github.com/BurntSushi/ripgrep/releases/download"
# (system, machine) → the release's target triple and its SHA-256. The Linux
# builds are the musl ones: statically linked, so the `rg` we carry does not
# ask the person's machine for a glibc newer than their own.
RIPGREP_TARGETS: dict[tuple[str, str], tuple[str, str]] = {
    ("Linux", "x86_64"): (
        "x86_64-unknown-linux-musl.tar.gz",
        "33e15bcf1624b25cdd2a55813a47a2f95dbe126268203e76aa6a585d1e7b149c",
    ),
    ("Linux", "aarch64"): (
        "aarch64-unknown-linux-musl.tar.gz",
        "800b1e7206afe799dfb5a6901f23147cfaabe0e52210538100f61e86e1740915",
    ),
    ("Darwin", "x86_64"): (
        "x86_64-apple-darwin.tar.gz",
        "af7825fcc69a2afc7a7aea55fc9af90e26421d8f20fe59df32e233c0b8a231c1",
    ),
    ("Darwin", "arm64"): (
        "aarch64-apple-darwin.tar.gz",
        "3750b2e93f37e0c692657da574d7019a101c0084da05a790c83fd335bad973e4",
    ),
    ("Windows", "AMD64"): (
        "x86_64-pc-windows-msvc.zip",
        "71b2fef860abe467217a538ff31de02f5258807c0129f771846f87bd029aafc5",
    ),
}


class PackError(Exception):
    """Something the build needs is not here."""


def rg_name() -> str:
    return "rg.exe" if sys.platform == "win32" else "rg"


def registry_modules() -> tuple[str, ...]:
    """The built-in shelf's modules. The registry scans them at run time —
    the bundler cannot see that — so each becomes a hidden import, and
    `pkgutil.iter_modules` finds them in the archive."""
    shelf = ROOT / "cli" / "agents"
    return tuple(
        sorted(
            f"cli.agents.{path.stem}"
            for path in shelf.glob("*.py")
            if not path.name.startswith("_")
        )
    )


def local_ripgrep() -> Path | None:
    """The `rg` this checkout already has: the venv's, else PATH's."""
    for candidate in (ROOT / ".venv" / "bin" / "rg", ROOT / ".venv" / "Scripts" / "rg.exe"):
        if candidate.is_file():
            return candidate
    found = shutil.which("rg")
    return Path(found) if found is not None else None


def fetch_ripgrep(into: Path) -> Path:
    """The official release for this platform, verified against the pinned
    digest. A mismatch is fatal: a binary we ship inside ours is one we have
    to be sure of."""
    key = (platform.system(), platform.machine())
    target = RIPGREP_TARGETS.get(key)
    if target is None:
        raise PackError(f"no pinned ripgrep for {key[0]}/{key[1]} — pack without --fetch-ripgrep")
    suffix, digest = target
    archive_name = f"ripgrep-{RIPGREP_VERSION}-{suffix}"
    url = f"{RIPGREP_URL}/{RIPGREP_VERSION}/{archive_name}"
    print(f"  fetching {archive_name}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload: bytes = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        raise PackError(f"{archive_name}: sha256 {actual}, expected {digest}")

    into.mkdir(parents=True, exist_ok=True)
    archive = into / archive_name
    archive.write_bytes(payload)
    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(into)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(into, filter="data")
    for path in into.rglob(rg_name()):
        path.chmod(0o755)
        return path
    raise PackError(f"{archive_name} holds no {rg_name()}")


def flags(ripgrep: Path | None) -> list[str]:
    """Everything PyInstaller is told, in one place."""
    arguments = [
        "--onefile",
        "--name",
        NAME,
        "--paths",
        str(ROOT),
        "--collect-all",
        "textual",
        "--collect-submodules",
        "mcp.client",
        "--collect-submodules",
        "mcp.server",
        "--collect-submodules",
        "mcp.shared",
        "--collect-all",
        "mcp_types",
        "--exclude-module",
        "mcp.cli",
        "--exclude-module",
        "typer",
    ]
    if ripgrep is not None:
        # `--add-binary src<sep>dest`, and the separator is the platform's.
        arguments += ["--add-binary", f"{ripgrep}{os.pathsep}."]
    for module in registry_modules():
        arguments += ["--hidden-import", module]
    return [
        *arguments,
        "--specpath",
        "build",
        "--workpath",
        "build/pyinstaller",
        "--distpath",
        "dist",
        ENTRY,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pack", description=__doc__)
    parser.add_argument(
        "--fetch-ripgrep",
        action="store_true",
        help="download the official ripgrep for this platform, not the venv's or PATH's",
    )
    parsed = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as scratch:
        try:
            ripgrep = fetch_ripgrep(Path(scratch)) if parsed.fetch_ripgrep else local_ripgrep()
        except PackError as error:
            print(f"pack: {error}", file=sys.stderr)
            return 1
        if ripgrep is None:
            print("pack: no ripgrep to carry — the binary will look for one on PATH")
        else:
            print(f"  carrying {ripgrep}")
        completed = subprocess.run(
            [sys.executable, "-m", "PyInstaller", *flags(ripgrep)], cwd=ROOT, check=False
        )
    if completed.returncode != 0:
        return completed.returncode
    built = ROOT / "dist" / (f"{NAME}.exe" if sys.platform == "win32" else NAME)
    print(f"\n  {built.relative_to(ROOT)}  ({built.stat().st_size // (1024 * 1024)} MB)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
