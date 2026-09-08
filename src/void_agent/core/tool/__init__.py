"""A model-callable capability, in two files:

- `tool` — the mechanism: `Tool` validates, gates, runs, serializes; `@tool`
- `gate` — the approval gate: `Approval` decides in code, `ensure_signed`
  asks the person and lets the call run or rejects it
"""

from void_agent.core.tool.gate import Approval, ensure_signed
from void_agent.core.tool.tool import Tool, tool

__all__ = ["Approval", "Tool", "ensure_signed", "tool"]
