"""The projections of a finished turn, split by audience:

- `accumulator` — wire events → the stored `parts` array (for the UI/store)
- `text`        — parts → plain text (titles, search)
- `context`     — parts → what the MODEL reads next turn (semantic resume)
- `content`     — parts → what the MODEL receives, attachments intact
- `attachment`  — the `file` part both model-facing projections read
"""

from void_agent.core.parts.accumulator import PartsAccumulator
from void_agent.core.parts.content import context_content
from void_agent.core.parts.context import TOOL_OUTPUT_CONTEXT_LIMIT, context_text
from void_agent.core.parts.text import parts_text

__all__ = [
    "TOOL_OUTPUT_CONTEXT_LIMIT",
    "PartsAccumulator",
    "context_content",
    "context_text",
    "parts_text",
]
