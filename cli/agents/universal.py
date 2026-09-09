"""The CLI's own agents.

`build_agent` is `universal`: the model, a plan, reflection, a question
— and, once the registry adds them, whatever MCP servers and skills the
process mounted. `chat` is the plain assistant with no tools, what the
registry answers with when there is nothing better to run: no provider,
no such agent. The example agent lives beside this one, in `weather.py`.
"""

from __future__ import annotations

from void_agent import HUMAN, Agent, Llm

SYSTEM = "You are void, a helpful assistant in a terminal. Answer in Markdown."

UNIVERSAL_SYSTEM = (
    "You are void, a helpful assistant in a terminal. Answer in Markdown.\n\n"
    "Your tools are whatever the user mounted — read their descriptions and use"
    " them; you may have none, in which case answer from what you know and say so"
    " when a task would need one.\n\n"
    "Keep your plan current with update_plan whenever a task takes more than one"
    " step. When a tool result surprises you or a step fails, use reflect before"
    " continuing.\n\n"
    "Judge for yourself as far as the facts allow. When a call fails, use what the"
    " error tells you and try again before concluding anything. Ask the user only"
    " when the way forward genuinely depends on them: ask_user with kind='choice'"
    " and the ways forward as options, or kind='input' for a missing fact, and"
    " carry on with the answer. Never end your turn with a question written in"
    " text.\n\n"
    "Some tools are marked as needing the user's signature: calling one shows them"
    " a card, and the call runs only if they sign it. That is normal — call the"
    " tool when the task needs it, and if they decline, say so and stop rather"
    " than trying another way around it."
)

MAX_STEPS = 80


def chat(llm: Llm) -> Agent:
    """The plain assistant: no tools, the model alone."""
    return (
        Agent(llm, "void", "the terminal assistant")
        .with_system(SYSTEM)
        .prompt(lambda history: list(history))
    )


def build_agent(llm: Llm) -> Agent:
    """the model, a plan, a question — and whatever MCP you mounted"""
    return (
        Agent(llm, "void", "the terminal assistant")
        .with_system(UNIVERSAL_SYSTEM)
        .with_max_steps(MAX_STEPS)
        .with_plan()
        .with_reflection()
        .prompt(lambda history: list(history))
        .tool(HUMAN)
    )
