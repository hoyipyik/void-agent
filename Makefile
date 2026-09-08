.DEFAULT_GOAL := help

.PHONY: help check test cli cli-build screenshots compile compile-verify

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

check: ## all quality gates: format, lint, types, tests
	uv run ruff format --check .
	uv run ruff check .
	uv run pyright
	uv run pytest -q

test: ## the test suite alone
	uv run pytest -q

cli: ## run the terminal UI on the host (uv); reads .env for the key
	uv run $(if $(wildcard .env),--env-file .env) python -m cli

# The flags live in the script, not here: Windows has neither `.venv/bin/rg`
# nor `sh`, and the release matrix (.github/workflows/ci.yml) builds there
# too. FETCH=1 carries the official ripgrep instead of this checkout's.
cli-build: ## pack the terminal UI into one binary: dist/void
	uv run --group build python scripts/pack.py $(if $(FETCH),--fetch-ripgrep)

screenshots: ## redraw the README's screenshots (screenshots/*.svg) on the configured provider; reads .env
	uv run $(if $(wildcard .env),--env-file .env) python scripts/screenshots.py

compile: ## wheel with core/ + providers/ as native .so (PY=3.12 picks the interpreter); see scripts/compile.py
	uv run python scripts/compile.py $(if $(PY),--python $(PY))

compile-verify: ## compile, then run the whole test suite against the installed wheel in a fresh venv
	uv run python scripts/compile.py $(if $(PY),--python $(PY)) --verify
