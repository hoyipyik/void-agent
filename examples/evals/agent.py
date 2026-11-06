"""The example suite's agent: a model that answers from the files its
mounted tools can read, with a plan and a way to ask the person. Its
builder has the shape every suite's has — the model and the mounted
tools in, the agent out."""

from __future__ import annotations

from collections.abc import Sequence

from void_agent import HUMAN, Agent, Llm, Tool

SYSTEM = """\
You answer questions about the orders in the files your tools can read. Answer only
from what they hold, and say so when they do not hold the answer. Be brief."""


def build(llm: Llm, tools: Sequence[Tool]) -> Agent:
    agent = (
        Agent(llm, "orders", "answers questions about the orders in a folder of files")
        .with_system(SYSTEM)
        .prompt(lambda history: list(history))
        .with_plan()
        .tool(HUMAN)
    )
    for tool in tools:
        agent.tool(tool)
    return agent
