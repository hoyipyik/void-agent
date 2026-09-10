# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); versioning: [SemVer](https://semver.org/).

## Unreleased

Everything — the framework has not been released yet.

- Core: the turn loop (`Answer` / `Ask` / raised error), `.tool(...)` as the
  single mounting verb (functions, workflows, sub-agents, `HUMAN`), plan and
  reflection as built-in tools, the
  Vercel-AI-SDK-compatible event stream, and `context_text` semantic resume.
- The model-facing projections keep a tool's output whole: `context_text`
  and `context_content` take `tool_output_limit=` for an application that
  wants a cap and cut the line there; core sets none. The CLI sets one:
  `context_tool_output_limit` in `~/.void/config.json`, 8,000 characters
  by default, 0 for none — an MCP server can hand back megabytes.
- The human as a channel: `agent.run(..., human=…)` makes an `Attendant`
  ambient for the whole tree; `ask_user` from any depth and a tool's
  `approval` gate put their question to it through the stream (`AskIssued`)
  and continue in place with the answer (`AskAnswered`) — no parent model
  relays anything. A gate asks for a signature and gets a boolean
  (`Attendant.approve`); `ask_user` gets words (`Attendant.answer`); the
  translation from UI vocabulary happens at the application's edge. No
  answer (nobody attending, or the attendant's patience spent) ends the run
  with the card open (`AskDropped` marks it) and every run above it the same
  way; the next message wakes the model. `Unanswered` is a BaseException, so
  no layer relays it and `except Exception` cannot swallow it.
- Core layout, one concern per file: leaves `errors`, `content`, `messages`,
  `ask` (a question as a value); packages `human/` (attendant, channel),
  `tool/` (tool, gate), `agent/` (agent, loop, dispatch, outcome, rules),
  `events/`, `parts/`, `llm/`, `builtins/` (plan, reflection).
- Providers: `AnthropicLlm`, `OpenAiLlm` (optional extras); `ScriptedLlm`
  for deterministic offline runs (steps may react to the transcript).
- `void_agent.mcp`: an MCP server's tools as `Tool`s, the approval declared
  by whoever mounts it. `void_agent.skills`: a folder of Markdown as tools
  whose whole effect is text in the transcript.
- The terminal UI (`cli/`, repo-only): the runtime in-process, the
  protocol rendered from parts, cards answered with the keys, the model's
  questions in the composer, attachments as `file` parts. Three agents:
  `universal` — the model, a plan, reflection, `ask_user`, void's own
  `toolbox` MCP server over the directory it started in, and whatever
  `~/.void/mcp.json` and `~/.void/skills` mount, switched in `/mcp` and
  `/skill`; `weather` — Open-Meteo behind five thin tools, the model as
  the scheduler; `dummy-weather` — the same on a scripted model and
  canned data, no key. `--agent module:function` mounts your own. Models from
  Anthropic, OpenAI-compatible endpoints, or a local Ollama, picked in
  `/model`. `make cli` runs it; `make cli-build` packs `dist/void`.
- Packaging: `uv build` is the wheel and the sdist. The wheel is pure
  Python (`py3-none-any`): one file for every OS, CPU and Python.
- CI: one workflow. `ci.yml` runs the gates and builds the wheel on every
  push; a `v*` tag packs the terminal UI (`void-agent-cli-<platform>`) for
  Linux (x86_64, arm64), macOS (Intel, Apple Silicon) and Windows — each
  started and spoken to before it is kept — and publishes the wheel and
  every binary as a GitHub release (the wheel to PyPI once trusted
  publishing is switched on). No sdist is published.
