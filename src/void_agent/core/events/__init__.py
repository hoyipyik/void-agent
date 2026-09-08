"""The event stream of one run, split by concern:

- `types`      — the vocabulary (frozen dataclasses, reserved kinds)
- `wire`       — the exact Vercel-protocol JSON (`to_wire`)
- `visibility` — what a caller sees of a subtree (`EventMode`)
- `sender`     — the shared producer handle (`EventSender`)
"""

from void_agent.core.events.sender import EventSender
from void_agent.core.events.types import (
    RESERVED_DATA_KINDS,
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
)
from void_agent.core.events.visibility import EventCategory, EventMode, category
from void_agent.core.events.wire import to_wire

__all__ = [
    "RESERVED_DATA_KINDS",
    "AgentEvent",
    "AskAnswered",
    "AskDropped",
    "AskIssued",
    "Error",
    "EventCategory",
    "EventMode",
    "EventSender",
    "Finish",
    "PlanUpdated",
    "Progress",
    "ReflectionMade",
    "Start",
    "StepStart",
    "TextDelta",
    "TextEnd",
    "TextStart",
    "ToolInputAvailable",
    "ToolInputStart",
    "ToolOutputAvailable",
    "ToolOutputError",
    "category",
    "to_wire",
]
