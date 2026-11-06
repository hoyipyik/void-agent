# CLAUDE.md

Evals: a set of questions put to an agent, every run written down whole,
for whoever reports on it later. A subpackage of the wheel, above core
like `mcp` and `skills` (`evals → core, mcp, providers`); nothing else in
`src/` imports it. `python -m void_agent.evals …`, or from this
repository's root `make evals ARGS="…"`, which reads `.env`. It needs the
`evals` extra and the extra of each provider a suite names.

What a user of it reads is `README.md` beside this file: the files a suite
is, how a case runs, the commands, a run's folder and a record's fields.
It is the reference: a change to any of those changes it in the same
commit. `skill/SKILL.md` is the skill for an agent that runs evals with
it (how to write cases, run, read and grade); it ships in the wheel.

## Layout

```text
suite.py   the config and the cases files as types; `select`; `expand`
           (`${NAME}`, `${NAME:-default}`, read when a persona is mounted)
load.py    the builder the config names: `package.module:function` with the
           suite's `paths` on sys.path, or `file.py:function` beside the config
model.py   the model, from [model]: the one place a provider's SDK is imported
person.py  the person a case with a `brief` is asked by, played by a model:
           answers `ask_user` in place and a question asked in words as
           their next message; silent when nothing was asked of them
mount.py   a persona's servers, mounted with its credentials; what each
           published (the catalog); its filters deciding offered and withheld
record.py  the three wrappers: `RecordingLlm` (each step: what it read new,
           what it returned), `CallLog` (each call: input, whole result,
           time, private failure detail), `EventLog` (every event, timed)
case.py    one case: mount, then each turn as a server runs one — the agent
           built afresh, the conversation so far handed in, the stream
           drained into parts with a server's end markers
run.py     the chosen cases side by side (`--jobs`), each `--repeat` times;
           the manifest; resume
store.py   a run's folder (below)
report.py  a run read back as Markdown; `show`'s step-by-step view
attach.py  a case's attached file as the `file` part a client sends
main.py    the commands
```

The program holds no suite and knows no trade: a suite is its own files,
wherever its owner keeps them, and the person's prompt names no business
(a suite says who they are, `[person] role`). Fixtures and examples use
made-up names only. `examples/evals/` is one suite, over void's own
toolbox server and four made-up orders
(`make evals ARGS="run examples/evals/eval.toml"`).

## Invariants

- Recording never changes what the agent does: a wrapper returns what it
  was given and re-raises what it caught. The agent is built per turn by
  the suite's own builder, from the same model and tools a server hands it.
- A credential is never written: headers and env are expanded only at
  mount, and the manifest names a persona's servers, not their settings.
- A record is written whole or not at all, so a stopped run resumes by
  asking only what has no record.
- Nothing is judged while a run runs. `expect` holds what a run can check
  by itself; whether an answer is right is the reader's, with `intent`.
- Tests drive it through `ScriptedLlm` and an in-memory MCP client
  (`tests/test_evals_*.py`); no test calls a model.
- `__init__` exports what a caller drives a run with itself: the suite
  as types, `run_cases`, `RunDir`, the report.
