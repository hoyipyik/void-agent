"""MACHINERY, not a concept: the OpenAI-compatible provider adapter.

OpenAI SDK types appear only here. The adapter speaks Chat Completions
streaming and rebuilds assistant turns from the parsed step — this API
needs no verbatim replay, so `raw` stays unset.

One provider quirk is quarantined here: some models echo their tool-call
arguments as message content in the same completion. A user-facing sentence
never starts with `{`, so when the request advertises tools, JSON-looking
content is held back until the stream ends, when it can be compared against
the captured tool calls: an actual echo (or an unparseable fragment of one)
is suppressed, while any other JSON-shaped answer is still delivered.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Sequence
from typing import Any, cast

from openai import AsyncOpenAI

from void_agent.core.content import ContentPart, ImageContent, TextContent
from void_agent.core.errors import Internal
from void_agent.core.events import EventSender, TextDelta, TextEnd, TextStart
from void_agent.core.llm import (
    AssistantStep,
    AssistantText,
    ModelStep,
    SystemText,
    ToolCall,
    ToolReturns,
    ToolSpec,
    TranscriptEntry,
    UserContent,
    UserText,
)


def _content(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.text}
    encoded = base64.b64encode(part.data).decode("ascii")
    if isinstance(part, ImageContent):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{part.media_type};base64,{encoded}"},
        }
    return {
        "type": "file",
        "file": {"filename": part.filename, "file_data": f"data:application/pdf;base64,{encoded}"},
    }


def _render(transcript: Sequence[TranscriptEntry]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for entry in transcript:
        match entry:
            case SystemText(text):
                messages.append({"role": "system", "content": text})
            case UserContent(content):
                messages.append({"role": "user", "content": [_content(part) for part in content]})
            case UserText(text):
                messages.append({"role": "user", "content": text})
            case AssistantText(text):
                messages.append({"role": "assistant", "content": text})
            case AssistantStep(step):
                message: dict[str, Any] = {"role": "assistant"}
                message["content"] = step.text or None
                if step.tool_calls:
                    message["tool_calls"] = [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.args, ensure_ascii=False),
                            },
                        }
                        for tool_call in step.tool_calls
                    ]
                messages.append(message)
            case ToolReturns(returns):
                messages.extend(
                    {
                        "role": "tool",
                        "tool_call_id": tool_return.call_id,
                        "content": tool_return.content,
                    }
                    for tool_return in returns
                )
    return messages


def _visible_held_text(held_text: str, tool_calls: Sequence[ToolCall]) -> str:
    """What of the held-back text may speak. Held text is whitespace or
    `{`-prefixed beside advertised tools: a complete JSON head equal to a
    captured call's arguments is the echo — only the words after it, if
    any, are real; an incomplete `{` head is an echo fragment; anything
    else is a genuine JSON-shaped answer, delivered whole."""
    head = held_text.lstrip()
    if not tool_calls or not head.startswith("{"):
        return held_text
    try:
        parsed, end = json.JSONDecoder().raw_decode(head)
    except json.JSONDecodeError:
        return ""
    if not any(tool_call.args == parsed for tool_call in tool_calls):
        return held_text
    return head[end:].strip()


class _PartialCall:
    __slots__ = ("arguments", "call_id", "name")

    def __init__(self) -> None:
        self.call_id = ""
        self.name = ""
        self.arguments = ""


class OpenAiLlm:
    """One configured model on an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model: str,
        *,
        client: AsyncOpenAI | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._client = client if client is not None else AsyncOpenAI()
        self._extra = dict(extra) if extra else {}

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": _render(transcript),
            "stream": True,
            **self._extra,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                }
                for spec in tools
            ]
        may_echo_arguments = len(tools) > 0

        text_id: str | None = None
        held_text = ""
        captured_text = ""
        ordered_calls: list[_PartialCall] = []
        open_by_index: dict[int, _PartialCall] = {}

        stream = cast("Any", await self._client.chat.completions.create(**request))
        async with stream:  # close the HTTP stream even on cancellation
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    captured_text += delta.content
                    if text_id is not None:
                        await events.send(TextDelta(id=text_id, delta=delta.content))
                    else:
                        held_text += delta.content
                        head = held_text.lstrip()
                        if head and not (may_echo_arguments and head.startswith("{")):
                            text_id = f"txt_{uuid.uuid4().hex}"
                            await events.send(TextStart(id=text_id))
                            await events.send(TextDelta(id=text_id, delta=held_text))
                            held_text = ""
                call_deltas: list[Any] = delta.tool_calls or []
                for call_delta in call_deltas:
                    partial = open_by_index.get(call_delta.index)
                    if partial is None or (
                        call_delta.id and partial.call_id and call_delta.id != partial.call_id
                    ):
                        # A fresh id on an already-used index is a NEW call —
                        # some compatible servers reuse index 0 for every call.
                        partial = _PartialCall()
                        open_by_index[call_delta.index] = partial
                        ordered_calls.append(partial)
                    if call_delta.id:
                        partial.call_id = call_delta.id
                    if call_delta.function is not None:
                        if call_delta.function.name:
                            partial.name = call_delta.function.name
                        if call_delta.function.arguments:
                            partial.arguments += call_delta.function.arguments

        tool_calls = tuple(self._parsed_call(partial) for partial in ordered_calls)

        if held_text:
            text = _visible_held_text(held_text, tool_calls)
            if text:
                text_id = f"txt_{uuid.uuid4().hex}"
                await events.send(TextStart(id=text_id))
                await events.send(TextDelta(id=text_id, delta=text))
        else:
            text = captured_text
        if text_id is not None:
            await events.send(TextEnd(id=text_id))

        return ModelStep(text=text, tool_calls=tool_calls)

    def _parsed_call(self, partial: _PartialCall) -> ToolCall:
        try:
            args: Any = json.loads(partial.arguments) if partial.arguments else {}
        except json.JSONDecodeError as error:
            raise Internal(f"parse {partial.name} arguments", error) from error
        if not isinstance(args, dict):
            raise Internal(
                f"parse {partial.name} arguments",
                TypeError(f"expected an object, got {type(args).__name__}"),
            )
        return ToolCall(
            call_id=partial.call_id or f"call_{uuid.uuid4().hex[:8]}",
            name=partial.name,
            args=cast("dict[str, Any]", args),
        )
