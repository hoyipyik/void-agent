"""The model boundary in one file: the step types one round-trip consumes
and produces, the transcript vocabulary a provider renders (framework
terms, no provider types), and the `Llm` protocol tying them together.
`AssistantStep` keeps a step verbatim (including opaque `raw`) so providers
that require exact continuation can replay it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from void_agent.core.content import ContentPart
from void_agent.core.events import EventSender
from void_agent.core.usage import Usage

# ── one round-trip ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call as the loop consumes it: arguments already parsed."""

    call_id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A capability as the model sees it advertised."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelStep:
    """The semantic result of one model round-trip. `usage` is what the
    provider said it cost, or None when it said nothing — the loop reports
    it, so a provider only has to carry it."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None
    usage: Usage | None = None


# ── the transcript a provider renders ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SystemText:
    text: str


@dataclass(frozen=True, slots=True)
class UserText:
    text: str


@dataclass(frozen=True, slots=True)
class UserContent:
    """Ordered text, image and PDF inputs in one user turn."""

    content: tuple[ContentPart, ...]


@dataclass(frozen=True, slots=True)
class AssistantText:
    """Replayed history: an assistant message with no live tool calls."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantStep:
    """A step this very run produced, kept verbatim so the provider can
    replay its own content (including `raw`) exactly."""

    step: ModelStep


@dataclass(frozen=True, slots=True)
class ToolReturn:
    call_id: str
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class ToolReturns:
    returns: tuple[ToolReturn, ...]


TranscriptEntry = SystemText | UserText | UserContent | AssistantText | AssistantStep | ToolReturns


# ── the protocol ──────────────────────────────────────────────────────────


class Llm(Protocol):
    """One model round-trip: render the transcript, stream text into the run
    as it arrives, return the parsed step. Implementations already carry
    their model choice — an agent consumes the pair, never a bare name."""

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep: ...
