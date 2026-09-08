"""The weather agent: five thin tools over Open-Meteo — the request each
one makes and what the model reads back — and the dummy that replays it
with no key and no network."""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest
from cli.weather import DEMO_QUESTION, build_agent, dummy, sky, weather_tools

from void_agent import Answer, EventSender, Message, Rejected, ScriptedLlm, Tool

TODAY = dt.date(2026, 9, 8)  # a Tuesday


def transport(routes: dict[str, dict[str, Any]], seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = routes.get(request.url.host)
        if body is None:
            return httpx.Response(429, text="Minutely API request limit exceeded")
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handle)


def named(tools: tuple[Tool, ...], name: str) -> Tool:
    return next(capability for capability in tools if capability.name == name)


GEOCODE = {
    "results": [
        {
            "name": "Springfield",
            "admin1": "Illinois",
            "country": "United States",
            "latitude": 39.8,
            "longitude": -89.65,
            "timezone": "America/Chicago",
            "population": 116250,
            "elevation": 179.0,  # not passed on: the model does not need it
        },
        {
            "name": "Springfield",
            "admin1": "Missouri",
            "country": "United States",
            "latitude": 37.22,
            "longitude": -93.3,
            "timezone": "America/Chicago",
        },
    ]
}

DAILY = {
    "timezone": "Asia/Taipei",
    "daily": {
        "time": ["2026-09-12", "2026-09-13"],
        "temperature_2m_max": [33.1, 31.0],
        "temperature_2m_min": [27.0, 26.4],
        "precipitation_sum": [0.0, 12.5],
        "precipitation_probability_max": [10, 80],
        "wind_speed_10m_max": [18.2, 25.0],
        "sunrise": ["2026-09-12T05:38", "2026-09-13T05:39"],
        "sunset": ["2026-09-12T18:04", "2026-09-13T18:03"],
        "weather_code": [1, 63],
    },
}


async def test_geocode_returns_every_candidate_trimmed_to_what_the_model_needs() -> None:
    seen: list[httpx.Request] = []
    tools = weather_tools(transport({"geocoding-api.open-meteo.com": GEOCODE}, seen))
    places: Any = await named(tools, "geocode").invoke(
        {"name": "Springfield", "count": 5}, EventSender()
    )
    assert seen[0].url.params["name"] == "Springfield"
    assert seen[0].url.params["count"] == "5"
    assert [place["admin1"] for place in places] == ["Illinois", "Missouri"]
    assert "elevation" not in places[0]
    assert "population" not in places[1]  # absent stays absent, not None


async def test_daily_asks_for_the_span_in_the_places_own_time_and_reads_it_back() -> None:
    seen: list[httpx.Request] = []
    tools = weather_tools(transport({"api.open-meteo.com": DAILY}, seen))
    forecast: Any = await named(tools, "daily").invoke(
        {
            "latitude": 25.05,
            "longitude": 121.53,
            "start_date": "2026-09-12",
            "end_date": "2026-09-13",
        },
        EventSender(),
    )
    params = seen[0].url.params
    assert (params["start_date"], params["end_date"], params["timezone"]) == (
        "2026-09-12",
        "2026-09-13",
        "auto",
    )
    assert forecast["timezone"] == "Asia/Taipei"
    sunday: Any = forecast["days"][1]
    assert (sunday["high_c"], sunday["low_c"], sunday["rain_chance_pct"]) == (31.0, 26.4, 80)
    assert (sunday["sky"], sunday["sunrise"]) == ("rain", "05:39")


async def test_the_archive_is_a_different_host_than_the_forecast() -> None:
    seen: list[httpx.Request] = []
    tools = weather_tools(transport({"archive-api.open-meteo.com": DAILY}, seen))
    await named(tools, "history").invoke(
        {
            "latitude": 25.05,
            "longitude": 121.53,
            "start_date": "2025-09-12",
            "end_date": "2025-09-13",
        },
        EventSender(),
    )
    assert seen[0].url.host == "archive-api.open-meteo.com"


async def test_an_unknown_place_and_a_refusal_are_readable_rejections() -> None:
    tools = weather_tools(transport({"geocoding-api.open-meteo.com": {"results": []}}, []))
    with pytest.raises(Rejected, match="no place called 'Atlantis'"):
        await named(tools, "geocode").invoke({"name": "Atlantis"}, EventSender())
    rate_limited = weather_tools(transport({}, []))
    with pytest.raises(Rejected, match="429"):
        await named(rate_limited, "current").invoke({"latitude": 0, "longitude": 0}, EventSender())


def test_the_sky_is_words_the_model_can_read() -> None:
    assert (sky(0), sky(63), sky(95)) == ("clear sky", "rain", "thunderstorm")
    assert sky(None) == "unknown"


def test_the_agent_carries_the_five_tools_the_plan_and_the_person() -> None:
    agent = build_agent(ScriptedLlm([]), today=TODAY)
    assert set(agent.tool_names) == {
        "update_plan",
        "reflect",
        "geocode",
        "current",
        "hourly",
        "daily",
        "history",
    }


async def test_the_dummy_replays_the_whole_trajectory_offline() -> None:
    """No key, no network: the scripted model resolves both places in one
    step, fetches both forecasts in one step, and answers with the numbers
    the canned data holds."""
    agent = dummy(ScriptedLlm([]), today=TODAY)  # the llm handed in is ignored
    result = await agent.run([Message.user(DEMO_QUESTION)])
    assert isinstance(result, Answer)
    assert "Osaka is the cooler" in str(result.value)
    assert "Sat 12 Sep" in str(result.value)  # the weekend after a Tuesday the 8th
