# void_agent.evals

Put a fixed set of questions to an agent, as each kind of user who would
ask them, and write down everything that happened: every step the model
took, what it read and what it returned, every tool call with its whole
result, every event. Then read the runs back.

It judges nothing while it runs. A case says what a good answer does
(`intent`) for whoever reads the run, and what a run can check by itself
(`expect`: the outcome, the tools called or avoided, how many calls).
Whether an answer is right is the reader's call: a person's, or an agent's
that grades the records.

It holds no suite. A suite is its owner's files, kept wherever the owner
keeps them; `examples/evals/` in the void-agent repository is one.

## Install

```sh
uv add "void-agent[evals,anthropic]"   # or [evals,openai]: the extra of each provider a suite names
python -m void_agent.evals --help
```

The `evals` extra brings the MCP SDK (servers are mounted through
`void_agent.mcp`). The provider SDKs are imported only when a suite names
that provider: `openai` (also for `ollama`), `anthropic`.

## The shortest path

1. Write the builder: a function that takes the model and the mounted tools
   and returns the agent (below).
2. Write `eval.toml`: the builder, the model, the MCP servers, the personas.
3. Write a cases file: one `[[cases]]` table per question.
4. `python -m void_agent.evals list eval.toml` shows what a run would ask;
   `mount eval.toml --persona NAME` shows the tools that persona is offered.
5. `python -m void_agent.evals run eval.toml` asks, writes
   `runs/<stamp>/`, and ends with `runs/<stamp>/report.md`.
6. Read `report.md`, then `show RUN KEY` for one case step by step, or the
   JSON records themselves.

## The builder

```python
from collections.abc import Sequence
from void_agent import Agent, HUMAN, Llm, Tool


def build(llm: Llm, tools: Sequence[Tool]) -> Agent:
    agent = Agent(llm, "shop", "answers questions about the shop's orders")
    agent.with_system("…").prompt(lambda history: list(history)).tool(HUMAN)
    for tool in tools:
        agent.tool(tool)
    return agent
```

- `llm` is the model the config names, already wrapped to record each step.
  Use it; never make another.
- `tools` are the tools the case's persona is offered, from every server,
  already wrapped to record each call. Add your own local tools beside them
  if the agent has any.
- It is called afresh for every turn, with the conversation so far handed
  in, the way a server builds an agent per request. Keep no state in it.
- The config names it as `package.module:function` (the suite's `paths` go
  on `sys.path` first) or `file.py:function` (a file beside the config).
- A builder that raises fails that case with `error.stage = "build"`; the
  others still run.

## eval.toml

Every path is relative to this file.

```toml
name = "shop"                       # the run's title; defaults to the file's stem
agent = "agent.py:build"            # or "package.module:build" with `paths`
paths = ["../src"]                  # put on sys.path before the builder is imported
cases = ["cases/manager.toml", "cases/clerk.toml"]
runs = "runs"                       # where runs are written; keep it out of git

[model]                             # the agent's model
provider = "anthropic"              # openai | anthropic | ollama
model = "claude-haiku-5-5"
effort = "high"                     # sent exactly as written; never checked here
extra = { cache_control = { type = "ephemeral" } }   # merged into every request

[context]                           # what a later turn reads of an earlier one
tool_output_limit = 2000            # characters of one tool result (omit: all of it)
tools = "mounted"                   # "all" | "mounted" | ["name", …]: whose calls are read back

[limits]
turn_seconds = 1800                 # one turn's limit; past it the turn is `cancelled`

[expect]                            # holds for every case's `expect`
uncounted = ["current_user"]        # calls that `max_calls` does not count

[person]                            # optional: a model plays the person of a case with a `brief`
provider = "anthropic"
model = "claude-haiku-5-5"
effort = "low"
rounds = 3                          # replies one case's person may give, at most
role = "a shop manager who asks about orders and stock"   # "You play <role>."

[servers.shop]                      # one MCP server: a url …
url = "${SHOP_MCP_URL:-http://localhost:8080/mcp}"
timeout = 60                        # seconds to mount, and for each call
signed = ["refund_*"]               # tools whose every call needs the person's signature

[servers.files]                     # … or a command, run as a subprocess
command = "uv"
args = ["run", "my-server", "${SUITE_DIR}/data"]
env = { DATA_MODE = "read-only" }

[personas.manager]                  # who a question is asked as
description = "sees every tool"
servers.shop.headers = { Authorization = "Bearer ${SHOP_TOKEN_MANAGER}" }

[personas.clerk]
description = "no sales figures"
exclude = ["sales_*"]               # and/or `only = [...]`: globs over tool names
servers.shop.headers = { Authorization = "Bearer ${SHOP_TOKEN_CLERK}" }
```

- Without any `[personas.*]`, there is one persona, `default`, with every
  server as written.
- A persona overrides a server's settings under `servers.<name>`
  (`headers`, `env`, …): that is where its credentials go.
- `${SUITE_DIR}` is this file's folder. Every other `${NAME}` (or
  `${NAME:-default}`) is read from the environment **when a persona is
  mounted**, so a missing token fails only that persona's cases, with a
  message naming the variable.
- `only` keeps the tools matching any of its globs; `exclude` then drops
  some. What is dropped is the persona's `withheld`, kept in each record.

## A cases file

Top-level `persona` and `tags` apply to every case in the file.

```toml
persona = "manager"
tags = ["orders"]

[[cases]]
id = "top-customers"                # letters, digits, . _ -; unique in the suite; names the record
ask = "Who bought the most this year?"
tags = ["ranking"]                  # added to the file's
intent = "Ranks customers by this year's total, per currency, and says the period."
expect = { outcome = "answered", uses = ["sales_total"], avoids = ["order_list"] }

[[cases]]
id = "follow-up"
turns = ["How much did Northwind buy?", { ask = "And on this order?", attach = ["order.pdf"] }]
intent = "The second turn reads the first and the attached order."

[[cases]]
id = "vague-price"
persona = "clerk"
ask = "what did we charge them for the box pc"
brief = "You mean Fabrikam, the French one. You want the unit price on their order."
intent = "Two customers bought it: asks which, then gives that one's price."

[[cases]]
id = "refund"
ask = "Refund order 1042."
replies = [true]                    # answers to the agent's questions, in order: words, or true/false to a signature
gap = "no refund tool yet"          # what the agent cannot do yet that this needs
expect = { max_calls = 2 }
```

| key | what it is |
| --- | --- |
| `id` | required; unique; the record's file name |
| `persona` | who asks; must be one of the config's personas |
| `ask` / `turns` | one message, or several: a string, or `{ ask, attach }` |
| `attach` | files beside the cases file, sent with the first turn (images, PDFs, text) |
| `replies` | scripted answers to `ask_user` (words) and to signatures (`true`/`false`), in order |
| `brief` | who is asking and what they want and know; the person plays it; the agent never sees it |
| `intent` | what a good answer does: for the reader, never checked |
| `gap` | what the agent cannot do yet that this needs |
| `tags` | for choosing cases (`--tag`) |
| `expect.outcome` | `answered`, `asked`, `failed`, `cancelled` or `error` |
| `expect.uses` | every one of these tools is called |
| `expect.uses_any` | at least one of these is |
| `expect.avoids` | none of these is |
| `expect.max_calls` | at most this many calls (`0`: an answer that needs none), besides `uncounted` |

## How a case runs

1. The persona's servers are mounted with its credentials; the catalog
   (what each server published, with fingerprints) and the offered and
   withheld tools are recorded. A mount that fails is `error`, stage `mount`.
2. Each turn: the agent is built afresh, handed the conversation so far,
   and run; its stream is drained into the parts a server would store.
   A later turn reads an earlier one as `[context]` says.
3. When the agent asks the person something:
   - a case with a `brief`, in a suite with a `[person]`: the person's
     model answers, in place for `ask_user`, or as the next message when the
     agent asked in words; it stays silent when nothing was asked; at most
     `rounds` replies. It sees the brief and the words of the chat, never a
     tool, a result or the agent's prompt. It never signs anything.
   - otherwise, `replies` answer in order;
   - with neither, the question ends the turn (`asked`), and the case's next
     turn, if any, is the answer.
4. Each turn ends `answered`, `asked` (it ended on a question nobody
   answered), `failed` (the agent's run raised) or `cancelled` (out of
   time). The case's outcome is its last turn's, or `error` when it could
   not run at all (`error.stage`: `mount`, `attach` or `build`).

## Commands

```text
python -m void_agent.evals list   CONFIG [choice]
python -m void_agent.evals mount  CONFIG --persona NAME
python -m void_agent.evals run    CONFIG [choice] [--jobs N] [--repeat N] [--label WORD]
                                  [--out DIR] [--resume RUN] [--model provider:model]
                                  [--effort E] [--turn-seconds S]
python -m void_agent.evals report RUN
python -m void_agent.evals show   RUN KEY
```

- A choice is any of `--case GLOB`, `--persona NAME`, `--tag TAG`, each
  repeatable: a case is chosen when it matches every kind given (any glob,
  any persona, any tag).
- `list` costs nothing; `mount` connects but asks no model; `run` calls the
  model with real keys.
- `run`: `--jobs` cases at once (default 2); each `--repeat` times (keys
  become `id~1`, `id~2`, …); `--label` goes in the folder's name; `--model`
  and `--effort` replace the suite's for this run.
- `--resume RUN` asks only the keys of RUN that have no record yet: a run
  stopped by Ctrl-C (it prints the command) or by a crash.
- `report RUN` rewrites `RUN/report.md`; `run` writes it at the end.
- `show RUN KEY` prints one case step by step: what each step read, its
  reasoning, each call with what it returned.
- Exit codes: `0` done; `1` no case chosen; `2` the suite or a mount is
  wrong (the message says what, in words a file can be fixed by); `130`
  stopped.

Keys and addresses come from the environment: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_HOST`, plus whatever `${NAME}`
the config names. Pass an env file with `uv run --env-file FILE python -m
void_agent.evals …`.

## A run's folder

```text
runs/<stamp>[-label]/
  run.json             what was run: suite, agent, model, context, limits,
                       personas (never credentials), the void-agent that asked
                       (`void_agent`: its version, and the commit a git install
                       came from or the checkout's revision), the commit, branch
                       and changes of the repository of each of the suite's
                       `paths`, the keys in order, start and end
  cases/<key>.json     one record per case (key = id, or id~n with --repeat)
  blobs/<sha256>.json  what records share: `specs`, `catalog`
  report.md
```

A record is written whole or not at all.

A record: `key id persona case started_at seconds outcome error usage
offered withheld specs catalog turns`, plus `person: {brief, moves}` for a
case with a person (a move: `at question reply seconds usage`; `question`
is null after a turn, the `ask_user` text in place).

- `case` is the case as the run read it, `expect` and `intent` included.
- `error`, when the case could not run: `{stage, text, detail}`, stage
  `mount`, `attach` or `build`.
- A turn: `user outcome answer ask asks error at seconds steps calls parts
  events`. `user.by` is `case` or `person`; `asks` holds each question put
  in place, with its answer.
- `steps[]`: `step at seconds read text calls usage reasoning raw error`.
  `read` is what the model had not read before: `{system}`, `{user: […]}`,
  `{assistant}` (an earlier turn as it was handed back),
  `{assistant_step: n}`, `{tool_returns: [{id, name, content}]}` (exactly
  what went back), `{user_text}` (the runtime's nudge). `raw` is the
  provider's output, opaque fields (`encrypted_content`, `signature`,
  `data`) given as their length.
- `calls[]`: `id name step source input status output chars error detail
  at seconds`. `status` is `ok`, `refused` (the tool's own words reached
  the model), `broken` (the model saw only an internal error; `detail` has
  the traceback), `held` (waiting on a question or a signature nobody
  gave), `error`, `cancelled`.
- `parts`: the turn as a server would store it, ending with
  `data-elapsed`, then `data-cancelled` or `data-error`.
- `events`: every event, `at` seconds into the case, a block's text deltas
  joined.

`report.md` has the run's header, a table per persona (outcomes, steps,
calls, seconds, tokens), a table per tool (calls, statuses, time, result
sizes, how many were cut by `tool_output_limit`), the arguments the model
filled in, the expectations missed, and every case: its turns, its calls,
its answer.

## Reading runs

- Start with `report.md`: *Expectations missed* lists what each case's
  `expect` caught.
- Grade an answer against its case's `intent`, using the calls' whole
  outputs in the record as the evidence of what the agent could know.
- To compare two versions of an agent, run the same cases on each (the
  same data behind the servers, `--label before` / `--label after`), and
  `--repeat` at least 2: the same question twice differs on its own.
- `run.json` (and the head of `report.md`) names the void-agent that
  asked and each repository's commit and whether it had changes: check it
  before comparing runs.

## Rules

- Recording never changes what the agent does: every wrapper returns what
  it was given and re-raises what it caught.
- A credential is never written: headers and env are expanded only at
  mount; the manifest names a persona's servers, not their settings.
- Runs hold whatever the servers returned. If that is real data, the runs
  folder stays out of git and out of anything published.
- Nothing is judged while a run runs.
