"""The approval gate: a side effect never runs on the model's word.

A tool may carry an `approval` — a function of the VALIDATED input that
returns a sentence when the call must be signed by a person first, or None
to let it run. The decision is made in code, before the handler. When a
signature is required, the call is put to whoever attends the run, right
here, and what comes back is a boolean: True, the handler runs and its
result is the call's result; False, the call is a readable rejection; no
decision, `Unanswered` unwinds everything above — nothing ran, and no caller
had to relay anything. Because the asking happens inside the tool, a
workflow three layers deep gets the same treatment as a root tool. Which
button or word meant "yes" is the application's business, never the gate's.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_core import to_jsonable_python

from void_agent.core.ask import Ask, Call, new_ask_id
from void_agent.core.errors import Rejected
from void_agent.core.events import EventSender
from void_agent.core.human import ask_signature

Approval = Callable[[Any], "str | Awaitable[str | None] | None"]


async def ensure_signed(
    approval: Approval, tool: str, validated: Any, events: EventSender
) -> None:
    """Take one call through its gate. Returns when the call may run: the
    approval asked for no signature, or the person signed. Raises `Rejected`
    when they declined; `Unanswered` passes through when no answer came."""
    reason = await _approval_reason(approval, validated)
    if reason is None:
        return
    if not await ask_signature(_signature_request(tool, reason, validated), events):
        raise Rejected(f"the user declined {tool}: {reason}")


async def _approval_reason(approval: Approval, validated: Any) -> str | None:
    """Why the call must be signed, or None; an approval may be sync or async."""
    reason = approval(validated)
    if inspect.isawaitable(reason):
        return await reason
    return reason


def _signature_request(tool: str, reason: str, validated: Any) -> Ask:
    """The gate's question as a card: the reason in the approval's own words,
    beside the exact call — validated input in its JSON shape, so what
    the person sees is what runs."""
    return Ask(
        ask_id=new_ask_id(),
        kind="approval",
        question=reason,
        options=None,
        payload=None,
        call=Call(tool=tool, input=to_jsonable_python(validated)),
    )
