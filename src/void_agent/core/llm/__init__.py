"""The model boundary, split by concern:

- `interface` — the round-trip types, the transcript entries a provider
  renders, and the one-method `Llm` protocol
- `scripted`  — the deterministic test/demo seam
"""

from void_agent.core.llm.interface import (
    AssistantStep,
    AssistantText,
    Llm,
    ModelStep,
    SystemText,
    ToolCall,
    ToolReturn,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    UserContent,
    UserText,
)
from void_agent.core.llm.scripted import (
    ScriptedLlm,
    ScriptedStep,
    call,
    last_tool_return,
    say,
    tool_call,
)

__all__ = [
    "AssistantStep",
    "AssistantText",
    "Llm",
    "ModelStep",
    "ScriptedLlm",
    "ScriptedStep",
    "SystemText",
    "ToolCall",
    "ToolReturn",
    "ToolReturns",
    "ToolSpec",
    "TranscriptEntry",
    "UserContent",
    "UserText",
    "call",
    "last_tool_return",
    "say",
    "tool_call",
]
