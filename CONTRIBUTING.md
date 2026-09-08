# Contributing

## Setup

```bash
uv sync        # Python 3.12+; installs the dev group
```

## The one gate

```bash
make check
```

That runs everything CI runs on a checkout: `ruff format --check`,
`ruff check`, `pyright` (strict), `pytest`. All of it is offline and
deterministic — no API key needed; agent behavior is tested through
`ScriptedLlm`.

CI also runs `make compile-verify` on every push: `core/` and `providers/`
compiled by Cython into a native wheel, the suite run against that wheel in
a fresh venv. A construct Cython cannot parse fails there, not at release —
the two rules it puts on the code are in CLAUDE.md under "Do not".

## Ground rules

- The core stays pydantic-only. Provider SDKs live in `providers/` behind
  optional extras.
- Every model-facing sentence (`core/agent/rules.py`, tool descriptions) is
  load-bearing: an edit is a behavior change and deserves a test.
- One concept per file; a new module's docstring states the concept AND why
  the boundary exists. Public API is re-exported from `void_agent`'s root.
- Failing test first; tests are named for behaviors
  (`test_a_valid_ask_preempts_sibling_side_effects`). `ScriptedLlm` and
  `CapturingLlm` are the seam — never mock our own modules.
- Open an issue before large changes.
