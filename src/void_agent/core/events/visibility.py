"""What a caller may observe of a subtree: `EventMode` narrows a registered
capability's events, and the strictest mode on the path to the root wins.
A subtree's text and step rhythm are its own voice — they never reach the
caller's stream. Its questions do: the attendant at the top answers them."""

from __future__ import annotations

import enum
from dataclasses import replace

from void_agent.core.events.types import (
    AgentEvent,
    AskAnswered,
    AskDropped,
    AskIssued,
    Error,
    Finish,
    PlanUpdated,
    Progress,
    ReflectionMade,
    Start,
    StepStart,
    TextDelta,
    TextEnd,
    TextStart,
    ToolInputAvailable,
    ToolInputStart,
    ToolOutputAvailable,
    ToolOutputError,
    UsageReported,
)


class EventCategory(enum.Enum):
    """The protocol-level kind of an event; every filter depends on it."""

    FRAMING = enum.auto()
    STEP = enum.auto()
    TEXT = enum.auto()
    TOOL_LIFECYCLE = enum.auto()
    PROGRESS = enum.auto()


def category(event: AgentEvent) -> EventCategory:
    """The event's protocol-level kind — deliberately without a wildcard."""
    match event:
        case Start() | Finish() | Error():
            return EventCategory.FRAMING
        case StepStart():
            return EventCategory.STEP
        case TextStart() | TextDelta() | TextEnd():
            return EventCategory.TEXT
        case ToolInputStart() | ToolInputAvailable() | ToolOutputAvailable() | ToolOutputError():
            return EventCategory.TOOL_LIFECYCLE
        # A subtree's usage is spent whoever spent it: it passes as progress,
        # so the account at the root is the whole tree's.
        case (
            UsageReported()
            | PlanUpdated()
            | ReflectionMade()
            | AskIssued()
            | AskAnswered()
            | AskDropped()
            | Progress()
        ):
            return EventCategory.PROGRESS


class EventMode(enum.IntEnum):
    """How much of a node's own events its caller observes.

    Variant order is the restriction order: nested `with_mode` calls keep the
    numerically largest — strictest — mode on the path to the root.
    """

    ALL = 0
    # Tool lifecycle and progress events pass with their full payloads; the
    # subtree's generated text and step rhythm are hidden — one voice, one
    # step counter, full trace.
    ACTIVITY = 1
    # `ACTIVITY` with tool output payloads redacted to None: what a subtree
    # produced belongs to the calling agent's own voice (or to a deliberate
    # `data-*` progress event), never to this stream.
    ACTIVITY_REDACTED = 2
    HIDDEN = 3


def _activity_visible(event: AgentEvent) -> AgentEvent | None:
    """The activity view of a subtree: lifecycle, progress, and questions
    pass, its generated text does not. Framing is emitted by the transport
    around the whole run, never inside it; a subtree's is not the run's —
    neither is its step rhythm (it would corrupt the root's step counter).
    A sub-agent's ask card DOES pass: the person attending the root answers
    it, and the answer returns to the sub-agent in place."""
    match category(event):
        case EventCategory.TOOL_LIFECYCLE | EventCategory.PROGRESS:
            return event
        case EventCategory.FRAMING | EventCategory.TEXT | EventCategory.STEP:
            return None


def _with_redacted_tool_output(event: AgentEvent) -> AgentEvent:
    if isinstance(event, ToolOutputAvailable):
        return replace(event, output=None)
    return event


def admit(mode: EventMode, event: AgentEvent) -> AgentEvent | None:
    match mode:
        case EventMode.ALL:
            return event
        case EventMode.HIDDEN:
            return None
        case EventMode.ACTIVITY:
            return _activity_visible(event)
        case EventMode.ACTIVITY_REDACTED:
            visible = _activity_visible(event)
            return None if visible is None else _with_redacted_tool_output(visible)
