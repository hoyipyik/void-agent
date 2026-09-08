"""`Agent`: configuration, input rendering, and running one turn.

Configuration chains at construction time (`with_system`, `prompt`,
`tool`, `output`, `with_plan`, ...) and mutates the SAME agent — not a
copy; a configured agent is then treated as immutable. Every run owns its
transcript and counters, so one agent serves any number of concurrent
runs.

`.tool(...)` registers a function tool, a workflow, or a sub-agent —
converted by `as_tool` into the same concrete `Tool` used for plain
functions, so the JSON boundary stays inside `Tool` — or the `HUMAN`
sentinel, which enables this model's `ask_user` calls. The agent owns the
tool registrations, human switch, output adapter, and output instructions.
`.output(Model, instructions=...)` defines what to return; the loop uses
`final_answer` as the model's submission protocol, not an ordinary handler.

`run(..., human=…)` makes an `Attendant` ambient for the whole tree: every
question below — this agent's, a sub-agent's, a gated tool's — reaches
that person directly and is answered in place; nobody answering ends the
run with the `Ask`, and a registered agent re-raises it so every run above
ends the same way. What happens inside a turn is the loop's business
(`agent/loop.py`).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from void_agent.core.agent.dispatch import ToolRegistration
from void_agent.core.agent.loop import Loop
from void_agent.core.agent.outcome import Answer, TurnResult
from void_agent.core.agent.rules import ASK_USER, FINAL_ANSWER, TOOL_SEQUENCING_RULE
from void_agent.core.ask import Ask, Human
from void_agent.core.builtins.plan import plan_tool
from void_agent.core.builtins.reflection import reflect_tool
from void_agent.core.content import TextContent
from void_agent.core.errors import Internal
from void_agent.core.events import EventMode, EventSender
from void_agent.core.human import Attendant, Unanswered, attended
from void_agent.core.llm import (
    AssistantText,
    Llm,
    SystemText,
    TranscriptEntry,
    UserContent,
)
from void_agent.core.messages import Message, Role
from void_agent.core.tool import Tool

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_RETRIES = 2


def json_prompt(input: Any) -> list[Message]:
    """The default prompt: the typed input, as JSON, in one user message."""
    return [Message.user(json.dumps(to_jsonable_python(input), ensure_ascii=False))]


class Agent:
    """A configured, reusable agent. See the module docstring."""

    def __init__(
        self,
        llm: Llm,
        name: str,
        description: str,
        *,
        input_type: type[Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.max_steps = DEFAULT_MAX_STEPS
        self.max_retries = DEFAULT_MAX_RETRIES
        self._llm = llm
        self._system = ""
        self._input_type = input_type
        self._tools: dict[str, ToolRegistration] = {}
        self._output_adapter: TypeAdapter[Any] | None = None
        self._output_instructions = ""
        self._asks_user = False
        self._prompt: Callable[[Any], Sequence[Message]] = json_prompt

    # ── construction-time configuration ──────────────────────────────────

    def with_system(self, system: str) -> Agent:
        self._system = system
        return self

    def with_max_steps(self, max_steps: int) -> Agent:
        self.max_steps = max_steps
        return self

    def with_max_retries(self, max_retries: int) -> Agent:
        self.max_retries = max_retries
        return self

    def with_plan(self) -> Agent:
        """Enable the built-in `update_plan` tool: the model keeps a live
        task-list projection (`PlanUpdated` → `data-plan`). Built-ins are
        chain-enabled — nothing to construct or inject."""
        return self.tool(plan_tool())

    def with_reflection(self) -> Agent:
        """Enable the built-in `reflect` tool: a durable projection of the
        model's self-assessment (`ReflectionMade` → `data-reflection`)."""
        return self.tool(reflect_tool())

    def prompt(self, render: Callable[[Any], Sequence[Message]]) -> Agent:
        """How this agent's typed input becomes the opening transcript.
        Context is input: history, task, everything arrives here."""
        self._prompt = render
        return self

    def output(self, output_type: type[Any], *, instructions: str = "") -> Agent:
        """Set the result type and task-specific guidance for submitting it.

        Instructions supplement the runtime's `final_answer` description;
        validation and completion remain enforced by the loop. Calling this
        method again replaces both the type and instructions.
        A model's docstring may also appear in its JSON schema description;
        use instructions here for task-specific guidance rather than duplicating it.
        The type must be object-shaped (a BaseModel, dataclass, or
        TypedDict) — every provider delivers tool arguments as a JSON
        object, so a bare scalar could never validate."""
        adapter: TypeAdapter[Any] = TypeAdapter(output_type)
        if adapter.json_schema().get("type") != "object":
            raise TypeError(
                f"agent `{self.name}` output type must be object-shaped (a BaseModel,"
                " dataclass, or TypedDict) — wrap the value in a model field"
            )
        self._output_adapter = adapter
        self._output_instructions = instructions.strip()
        return self

    @property
    def tool_names(self) -> tuple[str, ...]:
        """What this agent has registered, in registration order. The
        runtime's own `final_answer` and `ask_user` are not tools and are
        not among them."""
        return tuple(self._tools)

    def tool(self, capability: Tool | Agent | Human, *, mode: EventMode | None = None) -> Agent:
        """Register a tool, workflow, or sub-agent with its event visibility.
        HUMAN instead enables this model's `ask_user`; it does not change
        a child agent's setting or provide the attendant that answers.

        `mode` defaults per capability: `ALL` for function tools, `ACTIVITY`
        for sub-agents — a subtree's text and step rhythm are its own voice
        and would corrupt the root's stream; pass `mode=EventMode.ALL`
        explicitly to see the full nested trace."""
        if isinstance(capability, Human):
            self._asks_user = True
            return self
        if mode is None:
            mode = EventMode.ACTIVITY if isinstance(capability, Agent) else EventMode.ALL
        registered_tool = capability.as_tool() if isinstance(capability, Agent) else capability
        if registered_tool.name in (FINAL_ANSWER, ASK_USER):
            raise ValueError(f"`{registered_tool.name}` is reserved by the runtime")
        if registered_tool.name in self._tools:
            raise ValueError(
                f"tool names must be unique within agent `{self.name}`:"
                f" `{registered_tool.name}` is already registered"
            )
        self._tools[registered_tool.name] = ToolRegistration(registered_tool, event_mode=mode)
        return self

    def as_tool(self) -> Tool:
        """This agent as a model-callable `Tool`, without changing the
        registration API. The tool's output is the loop's answer as-is — a
        registered agent's answer goes straight into the parent's transcript.
        A question it could not get answered keeps unwinding: nothing in
        the sub-agent can go on without the answer, so its caller's run
        ends the same way, card and all."""
        if self._input_type is None:
            raise TypeError(f"agent `{self.name}` needs input_type to register as a tool")

        async def handler(input: Any, events: EventSender) -> Any:
            match await self.run(input, events):
                case Answer(value):
                    return value
                case Ask() as ask:
                    raise Unanswered(ask)

        handler.__annotations__["input"] = self._input_type
        return Tool(name=self.name, description=self.description, handler=handler)

    # ── one turn ──────────────────────────────────────────────────────────

    async def run(
        self,
        input: Any,
        events: EventSender | None = None,
        *,
        human: Attendant | None = None,
    ) -> TurnResult:
        """One turn: render the opening prompt, then loop until the model
        hands in (`Answer`) or a question goes unanswered (`Ask`). The
        caller of a channelled `EventSender` must drain its queue
        concurrently with this future.

        `human` attends the run: every question in the tree — this agent's,
        a sub-agent's, a gated tool's — goes to it and is answered in
        place. Omitted, the run inherits whoever attends the enclosing run
        (a registered sub-agent), or nobody: then the first question ends the
        turn with its card open on the session."""
        events = events if events is not None else EventSender()
        try:
            opening = list(self._prompt(input))
        except Exception as error:
            raise Internal("render agent prompt", error) from error
        transcript = self._opening_transcript(opening)
        loop = Loop(
            name=self.name,
            llm=self._llm,
            # A run keeps the registrations it started with: a later
            # `.tool(...)` on this agent must not reach a turn in flight.
            tools=self._tools.copy(),
            output_adapter=self._output_adapter,
            output_instructions=self._output_instructions,
            asks_user=self._asks_user,
            max_steps=self.max_steps,
            max_retries=self.max_retries,
        )
        if human is None:
            return await loop.run(transcript, events)
        with attended(human):
            return await loop.run(transcript, events)

    def _opening_transcript(self, opening: Sequence[Message]) -> list[TranscriptEntry]:
        system = f"{self._system}\n\n{TOOL_SEQUENCING_RULE}".strip()
        transcript: list[TranscriptEntry] = [SystemText(system)]
        for message in opening:
            if message.role is Role.USER:
                transcript.append(UserContent(message.content))
            else:
                (part,) = message.content  # assistant history is one text part: Message checks
                if not isinstance(part, TextContent):  # unreachable; narrows the type
                    raise TypeError("assistant history must be one text part")
                transcript.append(AssistantText(part.text))
        return transcript
