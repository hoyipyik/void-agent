"""What is the session's, and no Textual in it.

`store.py` is the session on disk — one JSON file of `parts`, the shape
the reference server's store keeps — and its model-facing projection;
`runner.py` is one turn, `agent.run` as a task and the loop that drains
its stream and its questions the way a server would; `asks.py` is the
desk, the questions a turn is waiting on and the one the composer is
answering; `attachments.py` is a file read for the person and made a
`file` part. The shell (`cli/shell.py`) is the screen these run under.
"""

from cli.session.store import (
    CONTEXT_MESSAGES,
    Session,
    SessionStore,
    SessionSummary,
    StoredMessage,
    Tally,
)

__all__ = [
    "CONTEXT_MESSAGES",
    "Session",
    "SessionStore",
    "SessionSummary",
    "StoredMessage",
    "Tally",
]
