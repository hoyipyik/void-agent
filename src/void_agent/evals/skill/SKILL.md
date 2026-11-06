---
name: void-agent-evals
description: Evaluate an agent built on void-agent with `void_agent.evals` — write a suite (a builder, eval.toml, cases files with personas, intents, briefs and expectations), run it with `python -m void_agent.evals`, and read or grade the recorded runs. Use when asked to test, eval, benchmark or compare an agent's answers or prompts, design questions for an agent, or read an eval run.
---

# Evaluating an agent with void_agent.evals

`void_agent.evals` asks a suite's questions, as each persona, and records
every step, call and event of every run. It judges nothing: you write what
a good answer does, and you (or a person) read the runs against it.

The full reference — every key of `eval.toml` and of a cases file, the
commands, a record's fields — is the README beside the module. Read it
before writing a suite:

```sh
python -c "import void_agent.evals as e, pathlib; print(pathlib.Path(e.__file__).parent / 'README.md')"
```

## 1. Know what you are testing

Before writing a case, find out:

- the agent's builder: the function that makes it from a model and tools
  (`build(llm, tools) -> Agent`). If the project only has a server that
  builds the agent inline, factor that out first: the eval must build it
  the same way the product does, or it tests something else.
- the tool sources (MCP servers), and what each kind of user may use:
  that is the personas, each with its own credentials and `only`/`exclude`.
- the data behind the servers. Runs you mean to compare need the same data
  behind them: a frozen copy, not a live system that changes between runs.

## 2. Write the cases

- One `[[cases]]` per question; `ask` for one message, `turns` for a
  conversation. Write questions the way users write them, typos and all,
  beside clean ones.
- `intent` is the grading key: what a right answer contains and does
  (the figures, the scope it states, what it must not do). Work the right
  answer out from the data before the run, not from the agent's output.
- `expect` only for what a run can check by itself: `outcome`, `uses`,
  `uses_any`, `avoids`, `max_calls`. Put tools that are called every turn
  (who-am-I, settings) in the config's `[expect] uncounted`.
- `brief` for a vague question: who asks, what they mean, what they know.
  With a `[person]` in the config, a model plays them and answers the
  agent's questions. Set `[person] role` to who the users are.
- `replies` instead of a brief for scripted answers: words for
  `ask_user`, `true`/`false` for a signature (`signed` tools).
- `gap` for what the agent cannot do yet; `tags` for choosing subsets.
- Keep a held-out set: cases written without looking at the agent's
  prompt, in their own file, that you never tune against.

## 3. Check before spending

```sh
uv run --env-file FILE python -m void_agent.evals list CONFIG          # parses, no model, no network
uv run --env-file FILE python -m void_agent.evals mount CONFIG --persona NAME   # tools offered / withheld
uv run --env-file FILE python -m void_agent.evals run CONFIG --case ONE-ID      # one real case
```

`list` exits 2 with a message naming the file and key when the suite is
wrong. `mount` shows exactly which tools each persona is offered, with
their fingerprints: compare it with what you meant.

## 4. Run

```sh
uv run --env-file FILE python -m void_agent.evals run CONFIG --jobs 4 --repeat 2 --label after
```

- `run` calls the real model: it costs money and time. Say how many cases
  × repeats you are about to ask before a large run.
- `--repeat 2` or more for anything you compare: the same question asked
  twice differs on its own, often by a tenth of the verdicts.
- Ctrl-C prints `--resume RUN`; resuming asks only what has no record.
- Runs land in the suite's `runs` folder. They hold whatever the servers
  returned: keep that folder out of git and out of anything published.

## 5. Read and grade

- `RUN/report.md` first: outcomes and calls per persona, each tool's
  statuses and result sizes (and how many results a later turn would cut),
  and *Expectations missed*.
- `show RUN KEY` for one case step by step: what the model read at each
  step, its reasoning, each call and its whole result.
- The records (`RUN/cases/<key>.json`) are the evidence. A wrong answer is
  one of: the tool returned the right data and the model misread it; the
  tool did not return it (wrong call, or the tool cannot); the question was
  ambiguous and the agent guessed instead of asking, or asked when it
  could have answered. Say which, with the call that shows it.
- Grade against `intent`, on a fixed scale (for example A–D), each repeat
  separately.
- Comparing two versions: same cases, same data, same model; check both
  `run.json` files name the commits you meant; grade blind (shuffle and
  relabel the two sides, keep the key apart); one grader for both sides.
  Report the held-out set separately.

## Do not

- Do not judge inside the agent or the suite: `expect` is mechanical,
  `intent` is for the reader.
- Do not put credentials in `eval.toml`: write `${NAME}` and pass an env
  file.
- Do not compare runs over data that changed between them, or across a
  different model or effort than you meant (`--model`, `--effort` override
  the suite's).
