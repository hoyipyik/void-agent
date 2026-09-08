#!/usr/bin/env python
"""The README's screenshots, drawn by the app itself on a live model.

    uv run --env-file .env python scripts/screenshots.py     (`make screenshots`)

Two scenes, each a real turn against the configured provider — a key in
the environment, as `void` itself reads it — written as SVG under
`screenshots/`. `VOID_HOME` is a temporary directory for the run (under ~, so the
welcome box reads naturally; removed after), so no session or config of
the person's is read or written, and the toolbox's
root is a scratch folder with one file in it. Nothing from the environment
reaches the file: the SVG is checked for the key before it is written.

  weather.svg   the weather agent on a two-city question: the plan, the
                parallel lookups, the table
  approval.svg  the universal agent about to edit a file: the signature
                card, held until the person answers — declined here
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from textual.pilot import Pilot
from textual.widgets import OptionList

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # `cli` is the checkout's, not an installed package
OUT = ROOT / "screenshots"
# Tall enough for the weather turn to keep its question in view; the card
# scene is shorter.
WEATHER_SIZE = (100, 70)
APPROVAL_SIZE = (100, 46)

WEATHER_QUESTION = "Which is cooler this weekend, Taipei or Osaka — and will either get rain?"
EDIT_REQUEST = "In README.md, change the title line to `# void` — nothing else."
README_BEFORE = "# a scratch project\n\nOne file, so the model has something to edit.\n"

# The toolbox starts `signed`; a person who works with it switches the
# reads on in `/mcp` and leaves the writes signed. So does this run, which
# is why the card that comes up is the edit, not the read before it.
READS_ON = (
    "read_file",
    "read_files",
    "list_directory",
    "directory_tree",
    "list_files",
    "search",
    "count_matches",
)


async def until(pilot: Pilot[None], condition: Callable[[], bool], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise RuntimeError("the scene never came up")
        await pilot.pause(0.1)


def secrets() -> tuple[str, ...]:
    return tuple(
        value
        for name, value in os.environ.items()
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(value) >= 8
    )


def write(name: str, svg: str) -> None:
    for secret in secrets():
        if secret in svg:
            raise RuntimeError(f"{name}: a secret from the environment is in the picture")
    target = OUT / f"{name}.svg"
    target.write_text(svg, encoding="utf-8")
    print(f"  {target.relative_to(ROOT)}  {len(svg) // 1024} KiB")


async def scenes(home: Path, project: Path) -> None:
    os.environ["VOID_HOME"] = str(home)
    # cli.main reads VOID_HOME at import, so it is imported after the override.
    from cli.agents import REGISTRY
    from cli.app import VoidApp
    from cli.config import load_config
    from cli.session import SessionStore
    from cli.widgets import AskCard

    config = load_config(os.environ, home / "config.json")
    if not config.configured():
        sys.exit("no provider in the environment: set ANTHROPIC_API_KEY or OPENAI_API_KEY")

    def app_for(agent: str) -> VoidApp:
        chosen = config.with_agent(agent)
        for name in READS_ON:
            chosen = chosen.with_tool_state(f"toolbox__{name}", "on")
        return VoidApp(
            REGISTRY.build_agent,
            store=SessionStore(home / "sessions"),
            config=chosen,
            config_file=home / "config.json",
            agents=REGISTRY,
            root=project,
        )

    def card_open(app: VoidApp) -> bool:
        return bool(app.query(AskCard)) and isinstance(app.focused, OptionList)

    print("weather: a live turn…")
    app = app_for("weather")
    async with app.run_test(size=WEATHER_SIZE) as pilot:
        await pilot.pause()
        await pilot.press(*WEATHER_QUESTION, "enter")
        await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
        await pilot.pause(0.5)
        write("weather", app.export_screenshot(title="void"))

    print("approval: a live turn, up to the card…")
    (project / "README.md").write_text(README_BEFORE, encoding="utf-8")
    app = app_for("universal")
    async with app.run_test(size=APPROVAL_SIZE) as pilot:
        await until(pilot, lambda: bool(app.bench.catalog), timeout=60)  # the toolbox is up
        await pilot.press(*EDIT_REQUEST, "enter")
        await until(pilot, lambda: card_open(app), timeout=180)
        await pilot.pause(0.5)
        write("approval", app.export_screenshot(title="void"))
        await pilot.press("2")  # decline: the file stays as it was
        await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
    assert (project / "README.md").read_text(encoding="utf-8") == README_BEFORE


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # The home is temporary but sits under ~, so the welcome box reads
    # `~/.void-…` rather than a /var/folders path; it is removed after.
    with (
        tempfile.TemporaryDirectory(prefix=".void-", dir=Path.home()) as home,
        tempfile.TemporaryDirectory() as project,
    ):
        asyncio.run(scenes(Path(home), Path(project)))


if __name__ == "__main__":
    main()
