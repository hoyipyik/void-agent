"""The weather agent replayed offline: the same agent on a scripted model
and canned data, so the whole protocol runs with no key and no network."""

from __future__ import annotations

import datetime as dt

from cli.agents.weather import build_agent as weather
from cli.agents.weather import canned_script, canned_transport
from void_agent import Agent, Llm, ScriptedLlm


def build_agent(llm: Llm, *, today: dt.date | None = None) -> Agent:
    """the weather agent replayed on a scripted model and canned data — no key, no network"""
    day = today or dt.date.today()
    return weather(ScriptedLlm(canned_script(day)), transport=canned_transport(day), today=day)
