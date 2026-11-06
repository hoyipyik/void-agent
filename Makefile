.DEFAULT_GOAL := help

.PHONY: help setup check test cli cli-build screenshots evals

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# The Python environment and nothing else: the app looks for its own
# providers (a key, a local Ollama) when it starts.
setup: ## first run: uv sync (dev + cli), .env from .env.example, ripgrep on PATH?
	@command -v uv >/dev/null || { echo "uv is missing: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	uv sync
	@test -f .env || { cp .env.example .env; echo "wrote .env from .env.example: fill in ONE provider's key (or leave it: the app asks on first start)"; }
	@command -v rg >/dev/null || test -x .venv/bin/rg || echo "note: ripgrep (rg) is neither on PATH nor in .venv; the CLI's search tool needs it"

check: ## all quality gates: format, lint, types, tests
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run pytest -q

test: ## the test suite alone
	uv run pytest -q

cli: ## run the terminal UI on the host (uv); reads .env for the key
	uv run $(if $(wildcard .env),--env-file .env) python -m cli

# A suite's paths are its own, so ARGS names the config wherever it is:
# make evals ARGS="run path/to/eval.toml --persona reader"
evals: ## ask a suite's questions and record every run (ARGS="run CONFIG …"; reads .env)
	uv run $(if $(wildcard .env),--env-file .env) python -m void_agent.evals $(ARGS)

# The flags live in the script, not here: Windows has neither `.venv/bin/rg`
# nor `sh`, and the release matrix (.github/workflows/ci.yml) builds there
# too. FETCH=1 carries the official ripgrep instead of this checkout's.
cli-build: ## pack the terminal UI into one binary: dist/void
	uv run --group build python scripts/pack.py $(if $(FETCH),--fetch-ripgrep)

screenshots: ## redraw the README's screenshots (screenshots/*.svg) on the configured provider; reads .env
	uv run $(if $(wildcard .env),--env-file .env) python scripts/screenshots.py
