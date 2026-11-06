"""The agent a suite names: a function of the model and the mounted tools
that returns an `Agent` — the shape a server's builder already has, so an
application's own builder is named as it is (`package.module:build`), or
from a file (`path/to/agent.py:build`). The suite's `paths` go on
`sys.path` first, so the builder's package imports as it does at home."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from void_agent.evals.case import Builder
from void_agent.evals.suite import SuiteError


def load_builder(spec: str, paths: Sequence[Path]) -> Builder:
    target, colon, attribute = spec.rpartition(":")
    if not colon or not target or not attribute:
        raise SuiteError(
            f"the agent is named as module:attribute or file.py:attribute, not {spec!r}"
        )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        if target.endswith(".py"):
            module_spec = importlib.util.spec_from_file_location(Path(target).stem, target)
            if module_spec is None or module_spec.loader is None:
                raise SuiteError(f"cannot load the agent from {target}")
            module = importlib.util.module_from_spec(module_spec)
            # Registered before it runs, as an import would: a dataclass in
            # it looks its own module up while it is being built.
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
    except (ImportError, OSError) as error:
        looked = ", ".join(str(path) for path in paths) or "sys.path"
        raise SuiteError(
            f"cannot import the agent {target} (looked in {looked}): {error}"
        ) from error
    builder = getattr(module, attribute, None)
    if not callable(builder):
        raise SuiteError(f"{target} has no function `{attribute}`")
    return cast("Builder", builder)
