"""MACHINERY, not a concept: the OpenAI Responses API adapter.

OpenAI SDK types appear only here. This is the API OpenAI's own reasoning
models take function tools on at any reasoning effort — Chat Completions
refuses tools beside one — so it is the adapter for OpenAI itself;
`openai.py` stays for the servers that speak only Chat Completions (Ollama
and its kin).

Stateless: `store` is off, so OpenAI keeps nothing and the transcript stays
the framework's. The reasoning comes back encrypted (`include`), and `raw`
keeps the response's output items verbatim, so a continuation replays them
— the reasoning, then the calls it led to — exactly as the API requires; a
step without one (scripted, foreign) is rebuilt from its text and calls.

Text streams as it arrives; the step's text and calls are read off the
final response, an incomplete one included (what it said stands). Its
usage counts the whole prompt in `input_tokens`, cached input inside it,
and the reasoning inside `output_tokens`.
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
from void_agent.core.usage import Usage


def _usage(reported: Any) -> Usage | None:
    if reported is None:
        return None
    details = getattr(reported, "input_tokens_details", None)
    return Usage(
        input=reported.input_tokens,
        output=reported.output_tokens,
        cache_read=getattr(details, "cached_tokens", None) or 0,
        cache_write=getattr(details, "cache_write_tokens", None) or 0,
    )


def _content(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextContent):
        return {"type": "input_text", "text": part.text}
    encoded = base64.b64encode(part.data).decode("ascii")
    if isinstance(part, ImageContent):
        return {
            "type": "input_image",
            "image_url": f"data:{part.media_type};base64,{encoded}",
            "detail": "auto",
        }
    return {
        "type": "input_file",
        "filename": part.filename,
        "file_data": f"data:application/pdf;base64,{encoded}",
    }


def _render(transcript: Sequence[TranscriptEntry]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in transcript:
        match entry:
            case SystemText(text):
                items.append({"role": "system", "content": text})
            case UserContent(content):
                items.append({"role": "user", "content": [_content(part) for part in content]})
            case UserText(text):
                items.append({"role": "user", "content": text})
            case AssistantText(text):
                items.append({"role": "assistant", "content": text})
            case AssistantStep(step):
                items.extend(_assistant_items(step))
            case ToolReturns(returns):
                items.extend(
                    {
                        "type": "function_call_output",
                        "call_id": tool_return.call_id,
                        "output": tool_return.content,
                    }
                    for tool_return in returns
                )
    return items


def _assistant_items(step: ModelStep) -> list[dict[str, Any]]:
    if step.raw is not None:
        return list(step.raw)
    items: list[dict[str, Any]] = []
    if step.text:
        items.append({"role": "assistant", "content": step.text})
    items.extend(
        {
            "type": "function_call",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.args, ensure_ascii=False),
        }
        for tool_call in step.tool_calls
    )
    return items


def _text(output: Sequence[dict[str, Any]]) -> str:
    return "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def _call(item: dict[str, Any]) -> ToolCall:
    name = str(item.get("name", ""))
    arguments = item.get("arguments") or ""
    try:
        args: Any = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as error:
        raise Internal(f"parse {name} arguments", error) from error
    if not isinstance(args, dict):
        raise Internal(
            f"parse {name} arguments",
            TypeError(f"expected an object, got {type(args).__name__}"),
        )
    return ToolCall(call_id=str(item["call_id"]), name=name, args=cast("dict[str, Any]", args))


class OpenAiResponsesLlm:
    """One configured model on OpenAI's Responses API, stateless."""

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
            "input": _render(transcript),
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            **self._extra,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_schema,
                    "strict": False,
                }
                for spec in tools
            ]

        text_id: str | None = None
        response: Any = None
        stream = cast("Any", await self._client.responses.create(**request))
        async with stream:  # close the HTTP stream even on cancellation
            async for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    if text_id is None:
                        text_id = f"txt_{uuid.uuid4().hex}"
                        await events.send(TextStart(id=text_id))
                    await events.send(TextDelta(id=text_id, delta=event.delta))
                elif event.type in ("response.completed", "response.incomplete"):
                    response = event.response
                elif event.type == "response.failed":
                    raise Internal("openai response", RuntimeError(event.response.error.message))
                elif event.type == "error":
                    raise Internal("openai response", RuntimeError(event.message))
        if text_id is not None:
            await events.send(TextEnd(id=text_id))
        if response is None:
            raise Internal("openai response", RuntimeError("the stream ended before its response"))

        output: list[dict[str, Any]] = [item.to_dict() for item in response.output]
        return ModelStep(
            text=_text(output),
            tool_calls=tuple(
                _call(item) for item in output if item.get("type") == "function_call"
            ),
            # An empty output must not be replayed, so it carries no raw and
            # the loop's own guard keeps the step out of the transcript.
            raw=output or None,
            usage=_usage(response.usage),
        )
