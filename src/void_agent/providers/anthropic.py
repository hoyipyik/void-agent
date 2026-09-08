"""MACHINERY, not a concept: the Anthropic provider adapter.

Anthropic SDK types appear only here. The adapter renders the framework
transcript to the Messages API, streams text into the run as it arrives,
and keeps `raw` as the verbatim response content so a continuation replays
thinking/tool blocks exactly as the API requires.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Sequence
from typing import Any

from anthropic import AsyncAnthropic

from void_agent.core.content import ContentPart, PdfContent, TextContent
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

DEFAULT_MODEL = "claude-opus-5"


def _content(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextContent):
        return {"type": "text", "text": part.text}
    source = {
        "type": "base64",
        "media_type": "application/pdf" if isinstance(part, PdfContent) else part.media_type,
        "data": base64.b64encode(part.data).decode("ascii"),
    }
    if isinstance(part, PdfContent):
        return {"type": "document", "title": part.filename, "source": source}
    return {"type": "image", "source": source}


def _render(transcript: Sequence[TranscriptEntry]) -> tuple[str, list[dict[str, Any]]]:
    """The framework transcript as (system, messages). An `AssistantStep`
    replays its verbatim `raw` content when the provider produced one —
    thinking blocks and signatures must return unmodified — and is rebuilt
    from the parsed step otherwise (a scripted or foreign step)."""
    system_texts: list[str] = []
    messages: list[dict[str, Any]] = []
    for entry in transcript:
        match entry:
            case SystemText(text):
                system_texts.append(text)
            case UserContent(content):
                messages.append({"role": "user", "content": [_content(part) for part in content]})
            case UserText(text):
                messages.append({"role": "user", "content": text})
            case AssistantText(text):
                messages.append({"role": "assistant", "content": text})
            case AssistantStep(step):
                messages.append({"role": "assistant", "content": _assistant_content(step)})
            case ToolReturns(returns):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_return.call_id,
                                "content": tool_return.content,
                            }
                            for tool_return in returns
                        ],
                    }
                )
    return "\n\n".join(system_texts), messages


def _assistant_content(step: ModelStep) -> Any:
    if step.raw is not None:
        return step.raw
    content: list[dict[str, Any]] = []
    if step.text:
        content.append({"type": "text", "text": step.text})
    for tool_call in step.tool_calls:
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.call_id,
                "name": tool_call.name,
                "input": tool_call.args,
            }
        )
    return content


class AnthropicLlm:
    """One configured model on the Anthropic Messages API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: AsyncAnthropic | None = None,
        max_tokens: int = 16000,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._client = client if client is not None else AsyncAnthropic()
        self._max_tokens = max_tokens
        self._extra = dict(extra) if extra else {}

    async def step(
        self,
        transcript: Sequence[TranscriptEntry],
        tools: Sequence[ToolSpec],
        events: EventSender,
    ) -> ModelStep:
        system, messages = _render(transcript)
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            **self._extra,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
                for spec in tools
            ]

        text_id: str | None = None
        async with self._client.messages.stream(**request) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                    and event.delta.text
                ):
                    if text_id is None:
                        text_id = f"txt_{uuid.uuid4().hex}"
                        await events.send(TextStart(id=text_id))
                    await events.send(TextDelta(id=text_id, delta=event.delta.text))
            message = await stream.get_final_message()
        if text_id is not None:
            await events.send(TextEnd(id=text_id))

        text = "".join(block.text for block in message.content if block.type == "text")
        tool_calls = tuple(
            ToolCall(call_id=block.id, name=block.name, args=dict(block.input))  # type: ignore[arg-type]
            for block in message.content
            if block.type == "tool_use"
        )
        # An empty content list must not be replayed — the API rejects an
        # empty assistant message — so an empty step carries no raw and the
        # loop's own guard keeps it out of the transcript.
        return ModelStep(text=text, tool_calls=tool_calls, raw=message.content or None)
