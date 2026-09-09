"""The weather agent: Open-Meteo behind five thin tools, the model as the
scheduler.

Ask what a forecast page cannot answer in one look — "which of Taipei,
Osaka and Singapore is coolest this weekend, and will any of them get
rain?", "how much warmer is London this week than the same week last
year?", "when tomorrow does the wind in Berlin drop below 20 km/h?". The
model resolves the places, fetches exactly what the question needs —
several places in one step, since those calls are independent — and does
the comparison itself. An ambiguous place is put to the person as a
choice, never guessed. No key: Open-Meteo is free.

The tools are thin on purpose: each is one request, its parameters the
tool's input, its answer trimmed to what a model can read. Everything
that looks like intelligence — which days, which hours, which of two
places, what the difference is — is the model's scheduling over them,
which is the framework's claim in one agent.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from void_agent import (
    HUMAN,
    Agent,
    Llm,
    ModelStep,
    Rejected,
    ScriptedStep,
    Tool,
    call,
    say,
    tool,
    tool_call,
)

GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 20.0
MAX_STEPS = 40

# WMO weather interpretation codes, as Open-Meteo returns them.
SKY: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    56: "freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def sky(code: object) -> str:
    return SKY.get(int(code), f"code {code}") if isinstance(code, int | float) else "unknown"


class Place(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="a city, town or region, as the person wrote it")
    count: int = Field(5, ge=1, le=10, description="how many candidates to return")


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float
    longitude: float


class Day(Point):
    date: str = Field(description="YYYY-MM-DD, in the place's own timezone")


class Span(Point):
    start_date: str = Field(description="YYYY-MM-DD, inclusive")
    end_date: str = Field(description="YYYY-MM-DD, inclusive")


def system(today: dt.date) -> str:
    return (
        "You are a weather analyst in a terminal. Today is"
        f" {today.isoformat()} ({today.strftime('%A')}).\n\n"
        "Places first: resolve every place with geocode before any lookup — the"
        " coordinates come from its result, so geocode and the lookup are dependent"
        " and belong in separate steps. Lookups that do not depend on each other —"
        " several places, several days — go in ONE step, as parallel calls.\n\n"
        "Dates: work in the place's own timezone, which geocode returns. 'This"
        " weekend' is the coming Saturday and Sunday; 'tonight' is this evening's"
        " hours; 'next week' is the Monday to Sunday after this one. Pick the tool by"
        " the grain of the question: current for now, hourly for part of a day, daily"
        " for whole days, history for the past.\n\n"
        "When geocode returns several plausible places and the question does not"
        " settle which — Springfield, San Jose, Cambridge — ask the person with"
        " ask_user kind='choice', the candidates as options, and carry on with their"
        " answer. Never guess a place.\n\n"
        "Keep the plan current with update_plan for anything beyond one lookup;"
        " reflect when a result surprises you. Do the comparisons and the arithmetic"
        " yourself and show the numbers you used. Answer in Markdown — a table when"
        " comparing places or days — and end with the answer to the question actually"
        " asked, in one sentence."
    )


def weather_tools(transport: httpx.AsyncBaseTransport | None = None) -> tuple[Tool, ...]:
    """The five tools, each one request. `transport` is the seam the tests
    use; left out, the requests go to Open-Meteo."""

    async def get(url: str, params: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as http:
            response = await http.get(url, params=params)
        if response.status_code >= 400:
            raise Rejected(f"Open-Meteo answered {response.status_code}: {response.text[:200]}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise Rejected("Open-Meteo answered with something other than an object")
        return cast("dict[str, Any]", payload)

    def coordinates(point: Point) -> dict[str, Any]:
        return {"latitude": point.latitude, "longitude": point.longitude, "timezone": "auto"}

    @tool(
        description=(
            "Places matching a name — up to `count`, each with coordinates, region, country"
            " and timezone. Several plausible ones means the question must say which, or"
            " the person must be asked."
        )
    )
    async def geocode(input: Place) -> list[dict[str, Any]]:
        found = await get(
            GEOCODING,
            {"name": input.name, "count": input.count, "language": "en", "format": "json"},
        )
        results = cast("list[dict[str, Any]]", found.get("results") or [])
        if not results:
            raise Rejected(f"no place called {input.name!r}")
        keys = ("name", "admin1", "country", "latitude", "longitude", "timezone", "population")
        return [
            {key: place.get(key) for key in keys if place.get(key) is not None}
            for place in results
        ]

    @tool(
        description=(
            "The weather right now at coordinates: temperature, feels-like, humidity,"
            " precipitation, cloud cover, wind and gusts, sky, and the local time."
        )
    )
    async def current(input: Point) -> dict[str, Any]:
        data = await get(
            FORECAST,
            {
                **coordinates(input),
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "precipitation,weather_code,cloud_cover,wind_speed_10m,wind_gusts_10m",
            },
        )
        now = data["current"]
        return {
            "time": now["time"],
            "timezone": data["timezone"],
            "temperature_c": now["temperature_2m"],
            "feels_like_c": now["apparent_temperature"],
            "humidity_pct": now["relative_humidity_2m"],
            "precipitation_mm": now["precipitation"],
            "cloud_cover_pct": now["cloud_cover"],
            "wind_kmh": now["wind_speed_10m"],
            "gusts_kmh": now["wind_gusts_10m"],
            "sky": sky(now["weather_code"]),
        }

    @tool(
        description=(
            "One day hour by hour at coordinates, in the place's local time: temperature,"
            " chance and amount of precipitation, wind, sky. Today to 16 days ahead."
        )
    )
    async def hourly(input: Day) -> dict[str, Any]:
        data = await get(
            FORECAST,
            {
                **coordinates(input),
                "hourly": "temperature_2m,precipitation_probability,precipitation,"
                "wind_speed_10m,weather_code",
                "start_date": input.date,
                "end_date": input.date,
            },
        )
        rows = data["hourly"]
        hours = [
            {
                "time": time[11:],
                "temperature_c": temperature,
                "rain_chance_pct": chance,
                "precipitation_mm": amount,
                "wind_kmh": wind,
                "sky": sky(code),
            }
            for time, temperature, chance, amount, wind, code in zip(
                rows["time"],
                rows["temperature_2m"],
                rows["precipitation_probability"],
                rows["precipitation"],
                rows["wind_speed_10m"],
                rows["weather_code"],
                strict=True,
            )
        ]
        return {"date": input.date, "timezone": data["timezone"], "hours": hours}

    @tool(
        description=(
            "Day by day between two dates at coordinates, today to 16 days ahead: high,"
            " low, precipitation and its top chance, strongest wind, sunrise, sunset, sky."
        )
    )
    async def daily(input: Span) -> dict[str, Any]:
        data = await get(
            FORECAST,
            {
                **coordinates(input),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset,weather_code",
                "start_date": input.start_date,
                "end_date": input.end_date,
            },
        )
        rows = data["daily"]
        days = [
            {
                "date": date,
                "high_c": high,
                "low_c": low,
                "precipitation_mm": rain,
                "rain_chance_pct": chance,
                "wind_max_kmh": wind,
                "sunrise": sunrise[11:],
                "sunset": sunset[11:],
                "sky": sky(code),
            }
            for date, high, low, rain, chance, wind, sunrise, sunset, code in zip(
                rows["time"],
                rows["temperature_2m_max"],
                rows["temperature_2m_min"],
                rows["precipitation_sum"],
                rows["precipitation_probability_max"],
                rows["wind_speed_10m_max"],
                rows["sunrise"],
                rows["sunset"],
                rows["weather_code"],
                strict=True,
            )
        ]
        return {"timezone": data["timezone"], "days": days}

    @tool(
        description=(
            "What the weather was, day by day between two past dates at coordinates —"
            " from 1940 to a few days ago: high, low, precipitation, strongest wind, sky."
        )
    )
    async def history(input: Span) -> dict[str, Any]:
        data = await get(
            ARCHIVE,
            {
                **coordinates(input),
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "wind_speed_10m_max,weather_code",
                "start_date": input.start_date,
                "end_date": input.end_date,
            },
        )
        rows = data["daily"]
        days = [
            {
                "date": date,
                "high_c": high,
                "low_c": low,
                "precipitation_mm": rain,
                "wind_max_kmh": wind,
                "sky": sky(code),
            }
            for date, high, low, rain, wind, code in zip(
                rows["time"],
                rows["temperature_2m_max"],
                rows["temperature_2m_min"],
                rows["precipitation_sum"],
                rows["wind_speed_10m_max"],
                rows["weather_code"],
                strict=True,
            )
        ]
        return {"timezone": data["timezone"], "days": days}

    return (geocode, current, hourly, daily, history)


def build_agent(
    llm: Llm,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    today: dt.date | None = None,
) -> Agent:
    """Open-Meteo: forecasts, hours, history — ask it anything about the weather"""
    agent = (
        Agent(llm, "weather", "answers weather questions from Open-Meteo")
        .with_system(system(today or dt.date.today()))
        .with_max_steps(MAX_STEPS)
        .with_plan()
        .with_reflection()
        .prompt(lambda history: list(history))
        .tool(HUMAN)
    )
    for capability in weather_tools(transport):
        agent = agent.tool(capability)
    return agent


# ── the same agent, replayed: no key, no network ──────────────────────────


def _weekend(today: dt.date) -> tuple[dt.date, dt.date]:
    saturday = today + dt.timedelta(days=(5 - today.weekday()) % 7 or 7)
    return saturday, saturday + dt.timedelta(days=1)


CANNED_PLACES: dict[str, dict[str, Any]] = {
    "taipei": {
        "name": "Taipei",
        "admin1": "Taipei",
        "country": "Taiwan",
        "latitude": 25.05,
        "longitude": 121.53,
        "timezone": "Asia/Taipei",
        "population": 7871900,
    },
    "osaka": {
        "name": "Osaka",
        "admin1": "Ōsaka",
        "country": "Japan",
        "latitude": 34.69,
        "longitude": 135.5,
        "timezone": "Asia/Tokyo",
        "population": 2592413,
    },
}

# (high, low, precipitation mm, top chance %, wind max, sunrise, sunset, code)
CANNED_DAYS: dict[str, list[tuple[float, float, float, int, float, str, str, int]]] = {
    "taipei": [
        (33.1, 27.0, 0.0, 10, 18.2, "05:38", "18:04", 1),
        (31.0, 26.4, 12.5, 80, 25.0, "05:39", "18:03", 63),
    ],
    "osaka": [
        (29.2, 22.1, 0.0, 5, 15.6, "05:41", "18:12", 0),
        (30.0, 22.8, 0.4, 20, 17.3, "05:42", "18:11", 2),
    ],
}


def canned_transport(today: dt.date) -> httpx.MockTransport:
    """Open-Meteo's two shapes, answered from the tables above."""
    saturday, sunday = _weekend(today)
    dates = [saturday.isoformat(), sunday.isoformat()]

    def handle(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.host.startswith("geocoding"):
            place = CANNED_PLACES.get(params.get("name", "").strip().lower())
            return httpx.Response(200, json={"results": [place] if place else []})
        latitude = float(params.get("latitude", "0"))
        key = min(CANNED_PLACES, key=lambda k: abs(CANNED_PLACES[k]["latitude"] - latitude))
        rows = CANNED_DAYS[key]
        return httpx.Response(
            200,
            json={
                "timezone": CANNED_PLACES[key]["timezone"],
                "daily": {
                    "time": dates,
                    "temperature_2m_max": [row[0] for row in rows],
                    "temperature_2m_min": [row[1] for row in rows],
                    "precipitation_sum": [row[2] for row in rows],
                    "precipitation_probability_max": [row[3] for row in rows],
                    "wind_speed_10m_max": [row[4] for row in rows],
                    "sunrise": [f"{date}T{row[5]}" for date, row in zip(dates, rows, strict=True)],
                    "sunset": [f"{date}T{row[6]}" for date, row in zip(dates, rows, strict=True)],
                    "weather_code": [row[7] for row in rows],
                },
            },
        )

    return httpx.MockTransport(handle)


def canned_script(today: dt.date) -> list[ScriptedStep]:
    """The trajectory a live model takes on the demo question, step by step:
    a plan; both places in one step; both forecasts in one step; the plan
    closed; the answer, with the numbers the canned data holds."""
    saturday, sunday = _weekend(today)
    span = {"start_date": saturday.isoformat(), "end_date": sunday.isoformat()}
    taipei, osaka = CANNED_PLACES["taipei"], CANNED_PLACES["osaka"]
    t_sat, t_sun = CANNED_DAYS["taipei"]
    o_sat, o_sun = CANNED_DAYS["osaka"]
    plan = [
        {"id": "1", "title": "resolve Taipei and Osaka", "status": "in_progress"},
        {"id": "2", "title": "fetch Saturday and Sunday for both", "status": "pending"},
        {"id": "3", "title": "compare highs and rain", "status": "pending"},
    ]
    done = [{**item, "status": "completed"} for item in plan]
    answer = (
        f"**This weekend ({saturday:%a %d %b} to {sunday:%a %d %b})**\n\n"
        "| | Sat high / low | Sun high / low | Rain |\n"
        "| --- | --- | --- | --- |\n"
        f"| Taipei | {t_sat[0]} / {t_sat[1]} °C | {t_sun[0]} / {t_sun[1]} °C |"
        f" Sunday: {t_sun[2]} mm, {t_sun[3]} % chance, {SKY[t_sun[7]]} |\n"
        f"| Osaka | {o_sat[0]} / {o_sat[1]} °C | {o_sun[0]} / {o_sun[1]} °C |"
        f" {o_sun[2]} mm at most, {o_sun[3]} % chance |\n\n"
        f"Osaka is the cooler of the two — about {t_sat[0] - o_sat[0]:.0f} °C below Taipei"
        f" on Saturday — and only Taipei is likely to get rain, on Sunday.\n\n"
        "*(This is the dummy agent: a scripted model over canned data. Pick"
        " `weather` in `/agent` for the real one.)*"
    )
    return [
        call("update_plan", {"items": plan}),
        ModelStep(
            text="",
            tool_calls=(
                tool_call("geocode", {"name": "Taipei"}),
                tool_call("geocode", {"name": "Osaka"}),
            ),
        ),
        call(
            "update_plan",
            {
                "items": [
                    {**plan[0], "status": "completed"},
                    {**plan[1], "status": "in_progress"},
                    plan[2],
                ]
            },
        ),
        ModelStep(
            text="",
            tool_calls=(
                tool_call(
                    "daily",
                    {"latitude": taipei["latitude"], "longitude": taipei["longitude"], **span},
                ),
                tool_call(
                    "daily",
                    {"latitude": osaka["latitude"], "longitude": osaka["longitude"], **span},
                ),
            ),
        ),
        call("update_plan", {"items": done}),
        say(answer),
    ]


DEMO_QUESTION = "Which is cooler this weekend, Taipei or Osaka — and will either get rain?"
