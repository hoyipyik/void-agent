"""The CLI's own agents.

`build_agent` is `universal`: the model, a plan, reflection, a question
— and, once the registry adds them, whatever MCP servers and skills the
process mounted. `chat` is the plain assistant with no tools, what the
registry answers with when there is nothing better to run: no provider,
no such agent. The example agent lives beside this one, in `weather.py`.

`universal` also knows when it is: the agent is built once per turn, so
its system prompt ends with the local date, weekday, time and zone read
at that moment.
"""

from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def clock(now: dt.datetime) -> str:
    """The moment as the model reads it: weekday, date, time, zone."""
    zone = now.tzinfo.key if isinstance(now.tzinfo, ZoneInfo) else now.tzname()
    return (
        "The user's local time when this turn began:"
        f" {now:%A, %Y-%m-%d %H:%M}, {zone} (UTC{now:%:z})."
    )


def local_now() -> dt.datetime:
    """Now, in this machine's zone: by its name (Asia/Shanghai) when the
    machine says it, else by its offset alone."""
    now = dt.datetime.now().astimezone()
    name = _zone_name()
    if name is None:
        return now
    try:
        return now.astimezone(ZoneInfo(name))
    except (ZoneInfoNotFoundError, ValueError):  # a POSIX rule, a name without tzdata
        return now


def _zone_name() -> str | None:
    """The machine's IANA zone: `TZ` when set, else where /etc/localtime
    links to. Windows has neither, and gets the offset."""
    if tz := os.environ.get("TZ", "").removeprefix(":"):
        return tz
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        return None
    return target.partition("zoneinfo/")[2] or None


def chat(llm: Llm) -> Agent:
    """The plain assistant: no tools, the model alone."""
    return (
        Agent(llm, "void", "the terminal assistant")
        .with_system(SYSTEM)
        .prompt(lambda history: list(history))
    )


def build_agent(llm: Llm, *, now: dt.datetime | None = None) -> Agent:
    """the model, a plan, a question — and whatever MCP you mounted"""
    return (
        Agent(llm, "void", "the terminal assistant")
        .with_system(f"{UNIVERSAL_SYSTEM}\n\n{clock(now or local_now())}")
        .with_max_steps(MAX_STEPS)
        .with_plan()
        .with_reflection()
        .prompt(lambda history: list(history))
        .tool(HUMAN)
    )
