"""The plan is a projection, never a state machine.

`update_plan` is an ordinary built-in tool, enabled with
`Agent.with_plan()`: the model rewrites its COMPLETE task list and a
`PlanUpdated` event carries it to the UI as a live checklist; the parts
(`data-plan`) keep the latest copy for the app to persist. The runtime
validates the shape and nothing else — no transition rules, no scheduler
reads it, and losing it never affects correctness. Scheduling stays where
it belongs: in the transcript (the model) or in code (a workflow function).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from void_agent.core.events import EventSender, PlanUpdated
from void_agent.core.tool import Tool


class PlanStatus(enum.StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    status: PlanStatus = PlanStatus.PENDING
    note: str | None = None


class PlanUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PlanItem]


PLAN_DESCRIPTION = (
    "Maintain your task list for this run. Submit the COMPLETE list every time — it "
    "replaces the previous one. Mark an item in_progress before starting it (at most "
    "one at a time), and mark it completed only in a step AFTER you have seen the "
    "result proving it finished — never in the same step as the action itself. Put "
    "durable artifacts the task produced (order ids, links, hard constraints) in its "
    "note so they survive into later turns."
)


def plan_tool() -> Tool:
    """The built-in `update_plan` tool. Stateless: the plan's home is the
    transcript (the call's own arguments), the event stream, and the parts.
    Last write wins."""

    async def update_plan(update: PlanUpdate, events: EventSender) -> str:
        await events.send(PlanUpdated(items=to_jsonable_python(update.items)))
        return "plan updated"

    return Tool(name="update_plan", description=PLAN_DESCRIPTION, handler=update_plan)
