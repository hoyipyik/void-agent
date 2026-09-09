"""The projections of a finished turn, split by audience:

- `accumulator` — wire events → the stored `parts` array (for the UI/store)
- `text`        — parts → plain text (titles, search)
- `context`     — parts → what the MODEL reads next turn (semantic resume)
- `content`     — parts → what the MODEL receives, attachments intact
- `attachment`  — the `file` part both model-facing projections read
- `usage`       — parts → the account (`data-usage`) as `Usage` values
"""

from void_agent.core.parts.accumulator import PartsAccumulator
from void_agent.core.parts.content import context_content
from void_agent.core.parts.context import context_text
from void_agent.core.parts.text import parts_text
from void_agent.core.parts.usage import total_usage, usage_of, usages

__all__ = [
    "PartsAccumulator",
    "context_content",
    "context_text",
    "parts_text",
    "total_usage",
    "usage_of",
    "usages",
]
