"""The agent and its loop, split by concern:

- `agent`    — `Agent`: tools, human setting, output type and instructions,
               `as_tool`, `run` (opens the transcript, attends, hands to the loop)
- `loop`     — model tool definitions, step, validate submission, ask, gather, stall
- `dispatch` — tool registrations (tool + event mode) and ordinary tool execution
- `outcome`  — how a turn ends (`Answer` | `Ask`)
- `rules`    — control-call names and default model instructions
"""

from void_agent.core.agent.agent import DEFAULT_MAX_RETRIES, DEFAULT_MAX_STEPS, Agent
from void_agent.core.agent.outcome import Answer, TurnResult
from void_agent.core.agent.rules import ASK_USER, FINAL_ANSWER, HELD_TOOL_OUTPUT

__all__ = [
    "ASK_USER",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_STEPS",
    "FINAL_ANSWER",
    "HELD_TOOL_OUTPUT",
    "Agent",
    "Answer",
    "TurnResult",
]
