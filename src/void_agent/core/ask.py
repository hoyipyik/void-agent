"""A question for the person, as a value — a leaf every layer may import.

`Ask` is the card: what goes out on the stream, what the attendant is
handed, and what a turn ends with when nobody answered. Two things raise
one. The model calls the synthetic `ask_user` tool — registering `HUMAN`
advertises it — and `AskInput` is what it fills in. A tool's approval
gate asks for a signature, and its card carries `Call`: the gated tool
and its validated input in JSON shape, written by the runtime so that
what the person sees is what runs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter


@dataclass(frozen=True, slots=True)
class Call:
    """A gated tool call as the human sees it: the tool and its validated
    input in JSON shape (defaults filled, coercions applied)."""

    tool: str
    input: dict[str, Any]

    def to_data(self) -> dict[str, Any]:
        return {"tool": self.tool, "input": self.input}


@dataclass(frozen=True, slots=True)
class Ask:
    """A question for the human. While someone attends the run it is
    answered in place and never becomes a turn's ending; it ends the turn
    only when nobody was attending or the wait was dropped. The session
    keeps the card (`data-ask`); the next message wakes the model, which
    reads what happened and carries on. `call` is set when the question
    came from a tool's approval gate: the exact call the human was shown."""

    ask_id: str
    kind: str
    question: str
    options: tuple[str, ...] | None
    payload: dict[str, Any] | None
    call: Call | None = None


def new_ask_id() -> str:
    return f"ask_{uuid.uuid4().hex}"


class AskInput(BaseModel):
    """The model-facing schema of the synthetic `ask_user` tool."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["input", "approval", "choice"] = "input"
    question: str
    options: list[str] | None = None
    payload: dict[str, Any] | None = None

    def to_ask(self) -> Ask:
        return Ask(
            ask_id=new_ask_id(),
            kind=self.kind,
            question=self.question,
            options=tuple(self.options) if self.options is not None else None,
            payload=self.payload,
        )


ask_adapter: TypeAdapter[AskInput] = TypeAdapter(AskInput)


class Human:
    """Marker registered with `agent.tool(HUMAN)`: the human as a capability —
    it gives the model the `ask_user` tool."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "HUMAN"


HUMAN = Human()
