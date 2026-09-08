"""Everything that can end a run early, classified by who may read it.

The classification IS the security boundary: `Rejected` and `Exhausted`
messages go back to the model (and the event stream) so it can route around
the failure; `Internal` detail stays private — the stream and the model see
only a generic notice, the logs see the cause.
"""

from __future__ import annotations


class RunError(Exception):
    """Base of the run-error taxonomy. Match on the subclasses."""


class Rejected(RunError):
    """A refused call, in model-readable words: bad input, unknown tool, a
    guard saying no. The model may correct itself and retry."""


class Exhausted(RunError):
    """A spent budget: step limit or stall budget. At a tool boundary the
    calling model may still route around it, so the message stays readable."""


class Internal(RunError):
    """A private failure. Neither the stream nor the model sees its detail."""

    def __init__(self, context: str, cause: BaseException) -> None:
        super().__init__(f"{context}: {cause}")
        self.context = context
        self.cause = cause


INTERNAL_PUBLIC_TEXT = "The request could not be completed. Please try again."


def public_text(error: RunError) -> str:
    """The stream- and user-facing words for a run-ending error. `Rejected`
    and `Exhausted` speak for themselves; an `Internal` never does —
    `str(Internal)` carries the private cause, so anything that leaves the
    process must go through here."""
    if isinstance(error, Internal):
        return INTERNAL_PUBLIC_TEXT
    return str(error)
