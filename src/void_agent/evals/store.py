"""A run's folder: `run.json` says what was run, `cases/<key>.json` holds
one record each, and `blobs/` what many records saw alike — the tools as
the model saw them, the catalog as each server published it — written
once under its SHA-256 and named from the records by that hash.

A record is written whole or not at all (a temporary file, then a
rename), so a run that is stopped leaves only records that landed, and
a resumed run asks the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from void_agent.evals.record import Record

# A record's fields that many records share: written to `blobs/` once.
SHARED = ("specs", "catalog")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


class RunDir:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path, label: str = "", now: datetime | None = None) -> RunDir:
        """A new folder under `root`, named for when it started."""
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        name = f"{stamp}-{label}" if label else stamp
        path = root / name
        suffix = 1
        while path.exists():
            suffix += 1
            path = root / f"{name}.{suffix}"
        path.mkdir(parents=True)
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> RunDir:
        if not (path / "run.json").is_file():
            raise FileNotFoundError(f"{path} is not a run: it has no run.json")
        return cls(path)

    def manifest(self) -> Record:
        path = self.path / "run.json"
        if not path.is_file():
            return {}
        return cast("Record", json.loads(path.read_text(encoding="utf-8")))

    def write_manifest(self, manifest: Record) -> None:
        _write(self.path / "run.json", _dumps(manifest))

    def written(self) -> set[str]:
        folder = self.path / "cases"
        return {path.stem for path in folder.glob("*.json")} if folder.is_dir() else set()

    def write(self, record: Record) -> None:
        kept = dict(record)
        for field in SHARED:
            if kept.get(field) is not None:
                kept[field] = self._blob(kept[field])
        _write(self.path / "cases" / f"{record['key']}.json", _dumps(kept))

    def record(self, key: str) -> Record:
        path = self.path / "cases" / f"{key}.json"
        return cast("Record", json.loads(path.read_text(encoding="utf-8")))

    def records(self) -> list[Record]:
        """Every record, in the order the run named its cases; any the
        manifest does not name, after them."""
        named: list[str] = list(self.manifest().get("keys", []))
        present = self.written()
        order = [key for key in named if key in present]
        order += sorted(present - set(order))
        return [self.record(key) for key in order]

    def blob(self, ref: str) -> Any:
        digest = ref.removeprefix("sha256:")
        return json.loads((self.path / "blobs" / f"{digest}.json").read_text(encoding="utf-8"))

    def _blob(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = self.path / "blobs" / f"{digest}.json"
        if not path.exists():
            _write(path, text)
        return f"sha256:{digest}"
