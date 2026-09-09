# CLAUDE.md

Textual terminal UI for void_agent, in-process. A client of the runtime's
event protocol: the rendering vocabulary is the event stream itself (text,
tool lifecycle, `data-*` cards), so a new tool needs no UI change.

## Stack

Python · Textual 8 (`textual>=8,<9`, pinned: majors move fast) · asyncio.
Runs with `python -m cli` (`make cli`); `make cli-build` packs `dist/void`
with PyInstaller through `scripts/pack.py`, which holds the flags because
the release matrix builds on Windows too and a Makefile's shell does not
go there — the MCP SDK's client, server and shared packages whole (it
resolves transports by name), textual whole, `rg` beside the bundle (the
official release, pinned by digest, when `--fetch-ripgrep` asks), and each
of the registry's modules as a hidden import (read from `CATALOG`).
The binary puts the working directory on `sys.path` so `--agent
module:function` finds a module beside the person, and
`scripts/smoke_toolbox.py` speaks one MCP round trip to it, since a
transport the bundler dropped fails at the first call, not at build time.
Tests in `tests/test_cli_*.py` drive the app through
`App.run_test()` with `ScriptedLlm` / `ScriptedHuman` — never a mocked
module of ours.

## Layout

Three layers, the dependencies pointing down. `main → app → shell →
widgets / screens → the rest`; nothing below `shell` imports `app` or
`shell`, and nothing outside `widgets/`, `screens/`, `shell.py`
and `app.py` imports Textual. `providers ← config ← llm` is
one way too.

- `main.py` the entry point (`VOID_HOME` moves the state dir;
  `--serve-toolbox` is the toolbox subprocess, handled before argparse) ·
  `app.py` `VoidApp`: what is the process's — the config and its file,
  the store, the registry, the bench and the shelf, Ollama, the OS
  clipboard — and the flows that write the config: `/model`, `/key`,
  `/agent`, `/mcp`, `/skill`, each a modal; the servers start when the
  shell says it is up (`Shell.Ready`), and again when `/mcp` opens on
  an `mcp.json` that changed since · `shell.py` `Shell`, the one
  screen the app shows: what is the session's — the log, the composer
  and its menu, the rewind, the attachments, the turn and its questions,
  and the session's own commands (`/session`, `/new`, `/clear`,
  `/attach`, `/paste`, `/detach`, `/status`, `/help`, `/quit`); the rest
  go up to the app; `app.shell` is how the tests reach it · `labels.py` `tilde`, `model_label`, `bar_label`, `tokens`, `usage_label`, `duration` · `commands.py` slash commands as `Spec`s: `matching` for
  the menu, `complete`, `parse` · `config.py` the provider, the agent
  and the marks: environment first (`ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` / `OLLAMA_MODEL` + `OLLAMA_HOST`; `VOID_AGENT` names
  the agent), `~/.void/config.json` fills in (written by the key prompt,
  `/key`, `/model`, `/agent`, `/mcp` and `/skill`, mode 0600) · `llm.py`
  `resolve_llm`, a `Config` as an `Llm` — the one file in the CLI that
  imports a provider's SDK, lazily · `agents/` which agent runs: `registry.py` a
  `Registry` of `universal` (`universal.py`: the default — the model, a plan,
  reflection, `ask_user`, and whatever MCP is mounted), `weather` and
  `dummy-weather` (`weather.py`, the example agent: Open-Meteo behind five thin tools — geocode,
  current, hourly, daily, history — the model as the scheduler; `dummy`
  is the same agent on a `ScriptedLlm` and a canned `httpx` transport, so
  the whole protocol runs with no key and no network) plus what
  `--agent` / `VOID_AGENT` mounted at start (`module:function`, imported at the door
  — a failure exits); `/agent` chooses among the mounted only;
  `Registry.build_agent(config)` is the per-turn factory · `session/`
  what is the session's, with no Textual in it: `store.py` sessions on
  disk (`~/.void/sessions/<id>.json`, parts verbatim, the server store's
  shape), their model-facing projection, and their account
  (`Session.tally`: the sum of the `data-usage` parts, the round-trips
  counted, the context the model read last) · `runner.py` `Turn`: one run
  as a task and the loop that drains its stream and its questions
  concurrently, folding events into parts and writing the transport
  markers (`data-elapsed`, how long the turn took, then `data-cancelled`
  or `data-error`), tested against `ScriptedLlm`
  directly · `asks.py` `Desk`: the questions a turn is waiting on and the
  one the composer answers in words — the application's edge, where an
  answer's shape is settled · `attachments.py` a file as a `file` part
  (images, PDFs, text; limits; `paths_in`, `mentions`) · `clipboard.py` the OS clipboard through
  osascript / wl-paste / xclip / PowerShell, read for a copied file or
  image and written on copy (pbcopy / wl-copy / xclip / PowerShell), the
  runner and the writer seams.
- `providers/` what `/model` lists: `catalog.py` the provider names and
  the cloud catalogue (name, blurb, the recommended default per keyed
  provider; `provider_of` reads a bare id) · `ollama.py` the local
  server's own list, read live (`/api/tags`, `/api/show` for what each
  can do) when the picker opens; `alias` is the short name shown,
  `find_installed` checks a bare name; the model itself is the OpenAI
  provider on `<host>/v1`, no key.
- `mcp/` the servers this process mounted: `spec.py` `read_servers`
  reads `~/.void/mcp.json` (the `mcpServers` shape every client uses),
  `builtin_server` is the toolbox as a spec, `open_server` the real
  thing, `signature_of` the gate for a marked tool · `bench.py` `Bench`
  starts them all on one keeper task — the SDK's transports must be
  closed by the task that opened them — sends each one's stderr to
  `~/.void/logs/<name>.log` (a stdio server logs to stderr, which in a
  TUI is the canvas being drawn on), discovers their tools as
  `<server>__<tool>`, and hands the registry the enabled ones, gated
  where the person marked them; a server the person turned off is never
  started, and toggling one remounts the bench.
- `toolbox/` void's own MCP server over one root given at launch:
  `root.py` the boundary (`resolve`, `inside`) and `clip` · `files.py`
  read, list, tree, write, edit, move · `search.py` ripgrep: search,
  count, files · `process.py` `run`, whose arguments are a list and
  never a shell string, so a signature card shows exactly the argv that
  runs · `server.py` mounts them on the root. The shell mounts it by
  re-running its own executable (`--serve-toolbox`), so it needs no
  node, no download and no interpreter of its own; `mcp.json` may name
  a server called `toolbox` to replace it. Nothing in it is gated —
  whether a call must be signed is `/mcp`'s answer, and the server is
  mounted `signed`.
- `widgets/` the protocol rendered, and the chrome around it: `turn.py`
  `TurnView.sync(parts)` walks the parts array and mounts a widget per
  part, live and replayed alike · `reply.py` the assistant's text with a
  dot in the gutter, the person's message · `fold.py` a line that folds
  open: `ToolChip` (`⏺ name(args)`, its result under `⎿`), `DataCard`
  (any other `data-*`) · `cards.py` the plan and the reflection ·
  `ask.py` the question card answered with the keys · `format.py` a
  payload as one line or in full · `welcome.py` the welcome box (the
  VoidAgent logo), the transcript's header, drawn once at the top of the
  log and left there, its agent and model lines kept current by
  `Shell.refresh_label` · `menu.py` the slash-command menu (`/` opens it,
  arrows move, Tab completes, Enter runs) · `prompt.py` the prompt frame
  (sign + composer), the status line (a spinner, the turn's activity —
  `activity(event)` — and how long this step has run, restarted at each
  `data-step`; the agent, the model, the context the last round-trip
  read and what the session has consumed on the right) ·
  `attachbar.py` the
  attachments waiting for the next message, the list theirs ·
  `panels.py` `Panel`, the `/help` and `/status` panels · `theme.py` the
  one Textual theme; every colour is quoted from it (`[$primary]`,
  `$success`) · `composer.py`
  the multi-line box: Enter sends; shift / alt / ctrl / cmd + Enter
  (Kitty-protocol terminals), `\` + Enter or a pasted newline adds a
  line; a pasted path attaches; ctrl+v / cmd+v ask for the OS clipboard;
  the keys the shell takes (menu, rewind, ↑ in an empty box) go up as
  `Navigate`.
- `screens/` the modals, each a list read with the keys: `chooser.py`
  the shape they share and the dialog CSS · `session.py`, `model.py`,
  `agent.py` the pickers · `key.py` a provider, then its key ·
  `switchboard.py` the list that stays open, its rows switches rather
  than a choice of one, with `mcp.py` and `skill.py` built on it.

## Invariants

- The runtime runs in-process: `agent.run(history, events, human=…)` on the
  app's own loop, the stream drained concurrently. No server is required;
  a remote mode over a server's SSE would be the same renderer behind a
  different transport.
- The CLI is a client, not an agent. The agent is any `build_agent(llm)`
  in the registry (`cli/agents/registry.py`); the default is `universal`.
  Mounting is a process-level act (`--agent module:function`
  at start, imported then); choosing is a session-level act (`/agent`,
  among the mounted only — a name that is not there is refused, never
  saved). The CLI hands every agent the session as history, whatever its
  own script expects. Nothing agent-specific lives in the CLI: a new
  tool, a new agent, a sub-agent all render through the protocol.
- Render from `parts`. `PartsAccumulator` folds the stream; the widgets are
  a function of the parts array, live and reloaded alike. One replay path.
- The session is the state: the assistant side of `history` is
  `context_text(parts)` (the user side, `context_content` when attachments
  exist), never raw streamed text.
- The account is read off the parts, never kept beside them. A turn's
  `data-usage` parts are summed into one muted trailer after everything
  the turn produced (`⏺ 3 steps · 12s · 5.4k in · 200 out`, the dot in
  the gutter like every line's, in `$accent`), live and replayed alike —
  never a line per round-trip in the flow. The time is the runner's
  `data-elapsed` marker, written like `data-cancelled`, so a replay says
  it too; the status line's stopwatch is the current step's, live only. The
  status line's right side says the context the last round-trip read
  and what the session has consumed, in and out together, live as each
  reports (`… · 12k ctx · 51k consumed`; the running sum is the
  session's fold plus the turn's events, and agrees with the fold once
  the turn is kept); `/status` gives the split (`Session.tally`). A
  model that reports nothing leaves no part, and nothing is estimated
  in its place.
- The person answers on the card. A `HumanChannel` attends the run;
  the card's first row / second row become the gate's boolean HERE, at
  the edge, never in core. The model's questions are answered in words:
  a choice on the card's rows (its last row, Other, opens the composer),
  an input in the composer. While a card is open it holds the keys —
  nothing else takes the focus.
- Models are chosen, not typed: `/model` lists `providers.catalog.CATALOG`, then what
  Ollama has installed that can call tools (asked when the picker
  opens); a bare `/model <id>` is the escape hatch for one the catalogue
  lacks. The catalogue is a menu, not a fence — never a validation.
- Ollama is extra, never core. The CLI looks for it itself — at the
  first start's prompt and whenever a picker opens, never in `make
  setup`, which is the Python environment and nothing else — and where
  it does not answer or has no model an agent can run on
  (`ollama.usable`), nothing offers it: no header in `/model`, no row in
  the key prompt; `/key ollama` says why. It is the one provider without
  a key: choosing one of its models is what configures it, and a bare
  Ollama name is checked against the installed list, since nothing else
  would catch a typo before the first turn. Its alias is a rule, not a
  guess (`ollama.alias`): the namespace and a quantisation-only tag
  dropped, the size kept (`27b`), an uncensored build marked `-U`.
- MCP and skills are mounted per process, chosen per session — the same
  shape as `--agent` and `/agent`. Every agent gets the same ones;
  nothing is configured per agent. `Registry.build_agent` adds them to
  whatever it built, so a mark lands on the next turn — except a
  server's, which starts or stops a process and so takes effect at once.
- The toolbox is void's own server, mounted by default over the
  directory the shell was started in, `signed` because it can write. It
  is a spec like any other — listed, marked and switchable in `/mcp` —
  and an `mcp.json` entry named `toolbox` replaces it: a default, never a
  fence.
- An unmarked tool takes its server's `"default"` from `mcp.json`, and a
  server that says nothing gets `signed`: adding one must never hand the
  model a set of ungated tools. `"default": "on"` is the person vouching
  for a server in their own file — a server's `readOnlyHint` is its word,
  readable through `describe`, never a decision. The board writes back
  only the rows that moved, so an untouched tool stays unmarked and
  `mcp.json` keeps meaning something.
- A switch takes effect where it is thrown. A tool's mark lands on the
  next turn (the agent is rebuilt every one); a server's lands at once —
  `McpPicker.ServerToggled` goes up to the app, which remounts and hands
  the board back what the bench holds, so its tools appear or vanish
  under the cursor. A board open while the first mount is still running
  fills in by itself when it lands.
- `/mcp` opens on `mcp.json` re-read. An entry added or changed since
  the servers were mounted restarts them, the board filling in when
  that lands; an unchanged file restarts nothing — looking is not a
  switch. The board heads with the servers up out of those named, the
  tools on out of those it lists, and the signed and off counts,
  recomputed on every switch.
- There is no panel of the agent's own tools: those are its builder's
  business, and mixing them in would blur who decided what.
- Copy lands on the OS clipboard. A drag selects in the log (Textual's
  own selection); ctrl+c copies through `App.copy_to_clipboard`, which
  the app overrides to write the OS clipboard (`Clipboard.write`) as
  well as Textual's OSC 52 escape — macOS Terminal ignores the escape
  and iTerm2 refuses it by default, `pbcopy` and its kin do not ask. ⌘C
  never reaches the app on most terminals — the terminal keeps it for
  its own (empty) selection — so the help says ctrl+c, and ⇧ + drag
  (⌥ in iTerm2) for the terminal's own selection, which ⌘C copies as
  usual. Mouse reporting is what costs the native selection, and it is
  kept: clicks fold chips, the wheel scrolls the log. The log takes no
  focus: a click or a drag on it leaves the keys where they were, on the
  composer or an open card.
- Colours come from `widgets/theme.py` by name; no widget carries a hex. The theme
  is an ANSI one: the background and the text are the terminal's own, so
  nothing paints a surface or tints with an alpha — a highlight is solid
  `$primary` with `$block-cursor-foreground`, a quiet border `$border-blurred`.
- Attachments are `file` parts (media type + base64 data URL).
  The CLI reads the bytes — a dragged path, the clipboard, an `@path`,
  `/attach` — and puts them in the part; core never opens a path. A
  terminal paste is text: an image comes from the OS clipboard (ctrl+v,
  `/paste`), never from the paste itself.
- Readable errors (`public_text`) reach the log and the next turn as a
  `data-error` part; Internal detail reaches neither.
- The rewind is the edit + redo: ↑ in an empty composer marks the last
  user message and loads its words; Enter truncates the session at that
  message (`Session.truncate`) and sends the replacement as a new turn;
  the log is drawn again from what is left. Nothing is edited in place.

## Do not

- Import anything of the CLI from `src/`. The framework never knows its
  clients.
- Add agent-specific step kinds: a new tool renders through the generic
  tool view, a new card through its `data-<name>` payload.
- Write a per-token store or widget update; deltas go through
  `Markdown.get_stream()`, which batches.
