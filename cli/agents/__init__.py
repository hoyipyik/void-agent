"""The built-in shelf: the agents the CLI ships, one per module, scanned
by the registry (`cli/registry.py`) like any folder of agents.

`universal.py` is the CLI's own agent — the model, a plan, reflection, a
question, and whatever MCP servers and skills the process mounted.
`weather.py` is the example: a real agent on a real API, the framework's
claim in one file, and the one to read before writing your own.
`dummy_weather.py` is the same agent replayed on a scripted model. An
agent is any module with a `build_agent(llm) -> Agent`; a module without
one is a helper and is not listed.
"""
