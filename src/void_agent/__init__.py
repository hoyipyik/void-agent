"""void_agent — a turn-based agent framework.

Session remembers · Turn runs · Model schedules · Tool asks · Human answers · Message wakes.

Reading the source? `core/__init__.py` is the map; `core/agent/agent.py`
and `core/agent/loop.py` are the overview, and every other module is a
leaf they pull in.
"""

import importlib.metadata

from void_agent.core.agent import (
    ASK_USER,
    FINAL_ANSWER,
    HELD_TOOL_OUTPUT,
    Agent,
    Answer,
    TurnResult,
)
from void_agent.core.ask import HUMAN, Ask, AskInput, Call, Human
from void_agent.core.builtins import (
    PlanItem,
    PlanStatus,
    PlanUpdate,
    ReflectInput,
)
from void_agent.core.content import ContentPart, ImageContent, PdfContent, TextContent
from void_agent.core.errors import (
    INTERNAL_PUBLIC_TEXT,
    Exhausted,
    Internal,
    Rejected,
    RunError,
    public_text,
)
from void_agent.core.events import (
    RESERVED_DATA_KINDS,
    AgentEvent,
    AskAnswered,
    AskDropped,
    AskIssued,
    Error,
    EventMode,
    EventSender,
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
    to_wire,
)
from void_agent.core.human import (
    Attendant,
    HumanChannel,
    Question,
    ScriptedHuman,
    Unanswered,
    attended,
)
from void_agent.core.llm import (
    AssistantStep,
    AssistantText,
    Llm,
    ModelStep,
    ScriptedLlm,
    ScriptedStep,
    SystemText,
    ToolCall,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    UserContent,
    UserText,
    call,
    last_tool_return,
    say,
    tool_call,
)
from void_agent.core.messages import Message, Role
from void_agent.core.parts import PartsAccumulator, context_content, context_text, parts_text
from void_agent.core.tool import Approval, Tool, tool

try:
    __version__ = importlib.metadata.version("void-agent")
except importlib.metadata.PackageNotFoundError:  # running from a checkout
    __version__ = "0.0.0.dev0"

__all__ = [
    "ASK_USER",
    "FINAL_ANSWER",
    "HELD_TOOL_OUTPUT",
    "HUMAN",
    "INTERNAL_PUBLIC_TEXT",
    "RESERVED_DATA_KINDS",
    "Agent",
    "AgentEvent",
    "Answer",
    "Approval",
    "Ask",
    "AskAnswered",
    "AskDropped",
    "AskInput",
    "AskIssued",
    "AssistantStep",
    "AssistantText",
    "Attendant",
    "Call",
    "ContentPart",
    "Error",
    "EventMode",
    "EventSender",
    "Exhausted",
    "Finish",
    "Human",
    "HumanChannel",
    "ImageContent",
    "Internal",
    "Llm",
    "Message",
    "ModelStep",
    "PartsAccumulator",
    "PdfContent",
    "PlanItem",
    "PlanStatus",
    "PlanUpdate",
    "PlanUpdated",
    "Progress",
    "Question",
    "ReflectInput",
    "ReflectionMade",
    "Rejected",
    "Role",
    "RunError",
    "ScriptedHuman",
    "ScriptedLlm",
    "ScriptedStep",
    "Start",
    "StepStart",
    "SystemText",
    "TextContent",
    "TextDelta",
    "TextEnd",
    "TextStart",
    "Tool",
    "ToolCall",
    "ToolInputAvailable",
    "ToolInputStart",
    "ToolOutputAvailable",
    "ToolOutputError",
    "ToolReturn",
    "ToolReturns",
    "ToolSpec",
    "TranscriptEntry",
    "TurnResult",
    "Unanswered",
    "UserContent",
    "UserText",
    "attended",
    "call",
    "context_content",
    "context_text",
    "last_tool_return",
    "parts_text",
    "public_text",
    "say",
    "to_wire",
    "tool",
    "tool_call",
]
