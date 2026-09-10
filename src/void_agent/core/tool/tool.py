"""A model-callable capability.

`Tool` wraps a typed async function: pydantic's `TypeAdapter` validates the
model's JSON arguments into the handler's annotated input type, and the
result is serialized back to JSON. Plain async functions, workflows, and
agents all enter the runtime through this shape, so an agent keeps
heterogeneous tools in one ordinary list.

The handler declares what it needs by signature: one positional parameter
for the typed input, and optionally a second parameter annotated EventSender
to publish progress or nested agent events. Its name is unrestricted, and
it may be keyword-only.

The model-facing schema is derived from that input type, unless the tool
declares one: a capability whose contract was written elsewhere — an MCP
server's tool descriptor — passes `input_schema` and annotates its handler
for whatever it accepts.

A tool may carry an approval gate (`core/tool/gate.py`): it reads the
validated input before the handler and, when the call must be signed, the
tool asks the person itself, right here — a signed call runs and its
result is this call's result; a declined one is a readable rejection; one
nobody answered unwinds as `Unanswered`, nothing having run.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from void_agent.core.errors import Internal, Rejected, RunError
from void_agent.core.events import EventSender
from void_agent.core.tool.gate import Approval, ensure_signed


def _input_adapter(
    handler: Callable[..., Awaitable[Any]], name: str
) -> tuple[TypeAdapter[Any], inspect.Parameter | None]:
    """Validate the input-plus-optional-EventSender contract at registration.
    Keep the event parameter's name and kind so invocation can bind it correctly."""
    parameters = list(inspect.signature(handler, eval_str=True).parameters.values())
    if not parameters:
        raise TypeError(f"tool `{name}` handler must take a typed input parameter")
    input_parameter = parameters[0]
    if input_parameter.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(f"tool `{name}` input parameter must accept a positional argument")
    if input_parameter.annotation is inspect.Parameter.empty:
        raise TypeError(f"tool `{name}` handler must annotate its input parameter")
    if len(parameters) > 2:
        raise TypeError(
            f"tool `{name}` handler accepts one input parameter and at most one EventSender"
        )
    event_parameter = parameters[1] if len(parameters) == 2 else None
    if event_parameter is not None:
        if event_parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"tool `{name}` EventSender parameter cannot be variadic")
        if event_parameter.annotation is not EventSender:
            raise TypeError(
                f"tool `{name}` second parameter must be annotated EventSender; "
                "put business arguments in the first input parameter"
            )
    return TypeAdapter(input_parameter.annotation), event_parameter


class Tool:
    """A concrete, reusable capability with its model-facing contract."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        handler: Callable[..., Awaitable[Any]],
        approval: Approval | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("tool name must not be empty")
        if not description.strip():
            raise ValueError(f"tool `{name}` must carry a model-facing description")
        self.name = name
        self.description = description
        self._handler = handler
        self._approval = approval
        self._adapter, self._event_parameter = _input_adapter(handler, name)
        # A tool whose contract was written elsewhere — an MCP server's
        # descriptor — declares it; everything else derives it from the type.
        self.input_schema: dict[str, Any] = (
            input_schema if input_schema is not None else self._adapter.json_schema()
        )

    async def invoke(self, raw_input: Any, events: EventSender) -> Any:
        """Validate, gate, run, serialize. Failures are classified: bad
        input and handler `Rejected` / `Exhausted` stay model-readable — a
        decline is one of them — and anything unexpected becomes a private
        `Internal`. A gate's question nobody answered is `Unanswered`, a
        BaseException that passes through here untouched."""
        try:
            validated = self._adapter.validate_python(raw_input)
        except ValidationError as error:
            raise Rejected(f"invalid {self.name} input: {error}") from error

        try:
            if self._approval is not None:
                await ensure_signed(self._approval, self.name, validated, events)
            if self._event_parameter is None:
                output = await self._handler(validated)
            elif self._event_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                output = await self._handler(validated, **{self._event_parameter.name: events})
            else:
                output = await self._handler(validated, events)
        except (RunError, asyncio.CancelledError):
            raise
        except Exception as error:
            raise Internal(f"tool {self.name}", error) from error

        try:
            return to_jsonable_python(output)
        except Exception as error:
            raise Internal(f"serializing {self.name} output", error) from error


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    approval: Approval | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Wrap a typed async function as a `Tool`.

    The name defaults to the function's name and the description to its
    docstring — but a missing description is a construction error, because
    the description is the model's routing table entry for this capability.
    `approval` decides in code: a function of the validated input that
    returns the reason a call must be signed by a human, or None to let it run.
    """

    def wrap(handler: Callable[..., Awaitable[Any]]) -> Tool:
        resolved_description = description or inspect.getdoc(handler) or ""
        return Tool(
            name=name or handler.__name__,
            description=resolved_description,
            handler=handler,
            approval=approval,
        )

    return wrap
