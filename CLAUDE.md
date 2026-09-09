# CLAUDE.md

A turn-based agent framework. The runtime is the point: the loop, the event
protocol, and the two projections (`parts` for storage, `context_text` for
the model) are ours; provider SDKs are quarantined in `providers/`.

## Layout

```text
src/void_agent/core/     the concepts, finely split (one concern per file):
                         leaves errors, content (text/image/PDF inputs), messages, ask (a question as a
                         value), usage (what a round-trip cost as a value);
                         packages human/ (attendant: who answers —
                         the protocol, the ambient contextvar, and the two
                         ready-made ones: `ScriptedHuman` and
                         `HumanChannel`, paired with a queue of `Question`s
                         replied to in place; channel: how a question
                         travels, `Unanswered`),
                         tool/ (tool: the mechanism; gate: decides in code,
                         asks the person itself), agent/ (agent, loop,
                         dispatch, outcome, rules), builtins/
                         (plan, reflection — shipped tools, chain-enabled
                         via `with_plan`/`with_reflection`, never injected),
                         events/ parts/ llm/ — pydantic only
src/void_agent/providers anthropic.py / openai.py — SDK types stay inside
src/void_agent/mcp       the bridge: an MCP server's tools as `Tool`s
                         (server.py mounts and discovers, result.py maps a
                         CallToolResult); the approval is declared by the
                         caller, never by the server; SDK types stay inside
src/void_agent/skills    a folder of instructions as tools: only the
                         description stays in context, the body arrives on
                         the call; knowledge, never capability
src/void_agent/__init__  the public API; users import from `void_agent`
cli/                     the terminal UI, in-process (repo-only, not
                         packaged): the `universal` agent — the model, a
                         plan, reflection, ask_user, void's own `toolbox`
                         server (read, search, edit, run over one root),
                         and whatever `~/.void/mcp.json` and
                         `~/.void/skills` mounted, marked in `/mcp` and
                         `/skill`; `weather` (cli/agents/weather.py: Open-Meteo
                         behind five thin tools, the model as the
                         scheduler) and `dummy-weather` (the same on a
                         scripted model and canned data);
                         the protocol rendered from parts, cards
                         answered in place, attachments as `file` parts;
                         the model from Anthropic, OpenAI, or a local
                         Ollama (its OpenAI-compatible `/v1`, the installed
                         list read live); `make cli` runs it, `make
                         cli-build` packs dist/void; see cli/CLAUDE.md
scripts/                 compile.py — the native wheel: core/ + providers/ as
                         .so, stubs beside them (`make compile`; its
                         docstring is the guide); pack.py — the CLI as one
                         binary (`make cli-build`); smoke_toolbox.py — one
                         MCP round trip against a packed binary;
                         screenshots.py — the README's screenshots, drawn
                         by the app on a live model (a key in the
                         environment; nothing of it reaches the file)
screenshots/             what scripts/screenshots.py drew; committed, since
                         the README shows them
tests/                   behavior-named, ScriptedLlm as the seam
```

Dependencies point one way inside core: `messages → content`,
`llm → content/events/errors/usage`, `agent → everything below`,
`builtins → tool/events`, `tool → ask/errors/events/human`,
`human → ask/events`, `events → ask/usage`, `parts → events/usage`; then `providers → core`,
`mcp → core`, `skills → core`; then `cli → void_agent`. Concept modules
never import providers, and the framework never imports the CLI: a client
that mounts an agent is an application above it.

## Invariants (do not break)

- `final_answer` is the output submission protocol, not a registered Tool.
  `.output(Model, instructions=...)` adds agent guidance to its description;
  guidance never replaces validation or the completion rules.
- A valid `final_answer` preempts BEFORE side effects: no sibling call
  runs, no transcript growth. `ask_user` is put to the attendant before
  any sibling side effect: answered, the answer is the call's result and
  the step goes on; unanswered, the turn ends there with the card open.
- The person attends the run through one channel (`run(..., human=…)`,
  ambient via a contextvar): a question from any depth — a sub-agent's
  `ask_user`, a gated tool three layers down — reaches them directly and
  its answer returns to the loop that asked, in place. No parent's model
  ever relays a question or an approval. An unanswered question is
  `Unanswered` (a BaseException, like cancellation): it unwinds every
  layer — workflows, sub-agent boundaries — and ends every run above it
  the same way; `except Exception` cannot swallow it.
- A gated call never runs on the model's word. The tool's `approval`
  decides in code whether the call must be signed and why; the gate
  (`ensure_signed`) puts that to the person and gets a SIGNATURE — a boolean:
  True runs the handler right there (its result is the call's result),
  False is a readable rejection, no decision unwinds — nothing ran, the
  card stays on the session marked dropped, and the next message wakes the
  model to carry on. Which button or word means "yes" is translated at
  the application's edge (the server or UI that mounts the agent), never in
  core. The model's `ask_user` is answered in words. Nothing is executed by
  the application on the model's or the session's word, and nothing is
  registered for approval's sake.
- An MCP server brings tools, never approvals. `void_agent/mcp` turns a
  server's descriptors into `Tool`s (`Tool(input_schema=…)` is the one
  seam: a contract written elsewhere), and whether a call must be signed
  is declared by whoever mounted it. A server's own hints
  (`readOnlyHint`, …) are its word — readable through `describe`, never a
  decision. Mounting is a lifecycle, not a call: the connection opens
  once and outlives the per-turn agent, so `McpServer` is an async
  context manager. An HTTP server's authentication is a `headers`
  mapping on the spec, nothing more.
- A skill brings knowledge, a tool brings capability, and the line does
  not move. `void_agent/skills` turns a folder of Markdown into tools
  whose whole effect is text in the transcript: no side effect, nothing
  to sign, no script run. That is what lets a skill be written by someone
  who never touches the code. There is no `with_skills()` on `Agent` — a
  skill has no event and no projection, and core never opens a path.
- `Rejected`/`Exhausted` text reaches the model and the stream; `Internal`
  detail reaches only logs — anything outbound goes through `public_text`.
- The wire protocol (`to_wire`, `PartsAccumulator`, `context_text`,
  `context_content`) is a contract — tests pin exact JSON/strings; change
  deliberately or not at all. A user's attachment is a UIMessage `file`
  part (media type + base64 data URL); `context_content` is the only place
  it becomes `ImageContent`/`PdfContent`. Core never fetches a URL or opens
  a path.
- `data-step` is stream-only. Reserved data kinds are refused at `Progress`
  construction — a tool can never forge provenance-carrying parts, nor
  the markers an application appends itself (`data-cancelled`,
  `data-error`, `data-elapsed`).
- The account is the provider's word, carried, never estimated: a
  `ModelStep` carries the `Usage` its provider reported (or None), the
  loop reports it as `UsageReported` the moment the step returns — before
  the step's calls run, so a crash never loses it — and it persists as a
  `data-usage` part the model never reads (`context_text` is silent on
  it). `input` is the whole prompt, cached tokens inside it, whatever the
  API's own split. Visibility passes it as progress, so a sub-agent's
  round-trips land on the root's account.
- Framing (`start`/`finish`/`error`) is the transport's job, never the
  runtime's.
- Events are observation: losing one must never affect correctness.
- Never swallow `asyncio.CancelledError` — cancellation outranks every
  other gather failure.

## Quality gates

`make check` runs them all: `uv run ruff format --check .` ·
`uv run ruff check .` · `uv run pyright` (strict) · `uv run pytest`. All
clean before any commit. `make compile-verify` builds the native wheel and
runs the framework's tests against it (not the CLI's: they exercise the
checkout's cli/, and they are POSIX-shaped — the release matrix verifies
the wheel on Windows too): CI runs it on every push, so run it locally
after any change Cython might not parse (see Do not). `make cli-build`
packs the binary; CI packs it on every push too and starts it, since the
imports a bundler cannot see break at start, not at build.

## TDD

Failing test first; tests are named for behaviors
(`test_a_valid_ask_preempts_sibling_side_effects`). `ScriptedLlm` /
`CapturingLlm` are the seam — never mock our own modules.

## Do not

- Commit `.env` (real keys live there; `.env.example` is the template).
- Commit or push unless asked.
- Inline a dynamic keyword mapping in the tool handler call: keep
  `event_arguments = {parameter.name: events}` followed by
  `handler(validated, **event_arguments)`. The inline dictionary triggered
  Cython's dynamic-key call optimization failure during `compile-verify`
  (`TypeError: sequence item 0: expected str instance, NoneType found`).
- Use PEP 695 generics in `src/` (`def f[T]…`, `class C[T]`, `type X = …`):
  Cython does not parse them. Write a `TypeVar` with `# noqa: UP047`
  (`core/human/channel.py`). A pydantic model that defines methods needs
  `ignored_types=(type(_method_probe),)` in its `model_config`
  (`core/ask.py`): compiled methods are not `FunctionType`.
