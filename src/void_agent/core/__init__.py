"""The framework's concept modules — pydantic only, no provider SDKs.

One concern per file, dependencies pointing one way:

- `errors`, `messages`, `ask`,
  `usage`                        leaves: the error taxonomy, the conversation
                                 vocabulary, a question for the person as a
                                 value, what a round-trip cost as a value
- `events/`                      the stream: types, wire, visibility, sender
- `parts/`                       the two projections: storage parts, model context
- `llm/`                         the model boundary and its scripted seam
- `human/`                       the person: who answers (attendant), how a
                                 question travels (channel)          — Human answers
- `tool/`                        the mechanism (tool) and the approval gate
                                 that asks the person itself (gate)  — Tool asks
- `builtins/`                    the tools the framework ships
- `agent/`                       the configured agent and its loop:
                                 agent, loop, dispatch, outcome, rules

The public API is re-exported at the package root; import from
`void_agent`, not from here, unless you need an internal seam.
"""
