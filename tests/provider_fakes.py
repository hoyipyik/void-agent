"""In-memory SDK streams shared by provider contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class AnthropicStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events = events
        self._final = final

    async def __aenter__(self) -> AnthropicStream:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def __aiter__(self) -> Any:
        async def iterate() -> Any:
            for event in self._events:
                yield event

        return iterate()

    async def get_final_message(self) -> Any:
        return self._final


class AnthropicClient:
    def __init__(self, events: list[Any], final: Any) -> None:
        self.requests: list[dict[str, Any]] = []
        self._events = events
        self._final = final

    @property
    def messages(self) -> Any:
        outer = self

        class Messages:
            def stream(self, **kwargs: Any) -> AnthropicStream:
                outer.requests.append(kwargs)
                return AnthropicStream(outer._events, outer._final)

        return Messages()


def text_delta(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def final_message(*blocks: Any, usage: Any = None) -> Any:
    return SimpleNamespace(content=list(blocks), usage=usage)


def anthropic_usage(
    input_tokens: int,
    output_tokens: int,
    *,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> Any:
    """The SDK's usage shape: the cached counts are None when the request
    used no cache at all."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(call_id: str, name: str, input: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=input)


def chunk(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def usage_chunk(prompt_tokens: int, completion_tokens: int, *, cached: int | None = None) -> Any:
    """The last chunk when usage was asked for: no choices, the usage
    alone; `prompt_tokens_details` is absent on many compatible servers."""
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=None) if cached else None
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=details,
    )
    return SimpleNamespace(choices=[], usage=usage)


def call_delta(
    index: int, call_id: str | None = None, name: str | None = None, arguments: str | None = None
) -> Any:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class OpenAiStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self) -> OpenAiStream:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        self.closed = True
        return False

    def __aiter__(self) -> Any:
        async def iterate() -> Any:
            for item in self._chunks:
                yield item

        return iterate()


class OpenAiClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.streams: list[OpenAiStream] = []
        self._chunks = chunks

    @property
    def chat(self) -> Any:
        outer = self

        class Completions:
            async def create(self, **kwargs: Any) -> Any:
                outer.requests.append(kwargs)
                stream = OpenAiStream(outer._chunks)
                outer.streams.append(stream)
                return stream

        return SimpleNamespace(completions=Completions())


class OpenAiItem:
    """An output item as the SDK hands it: `to_dict` is the API's own JSON."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def output_text_delta(delta: str) -> Any:
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def response_done(
    *items: dict[str, Any], usage: Any = None, kind: str = "response.completed"
) -> Any:
    """The event that closes a response, `response.completed` unless told."""
    response = SimpleNamespace(output=[OpenAiItem(item) for item in items], usage=usage)
    return SimpleNamespace(type=kind, response=response)


def response_failed(message: str) -> Any:
    error = SimpleNamespace(code="server_error", message=message)
    return SimpleNamespace(
        type="response.failed", response=SimpleNamespace(output=[], usage=None, error=error)
    )


def response_usage(input_tokens: int, output_tokens: int, *, cached: int = 0) -> Any:
    """`input_tokens` is the whole prompt, the cached share inside it."""
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=None)
    return SimpleNamespace(
        input_tokens=input_tokens, output_tokens=output_tokens, input_tokens_details=details
    )


def message_item(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call_item(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def reasoning_item(encrypted: str) -> dict[str, Any]:
    return {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": encrypted}


class OpenAiResponsesClient:
    def __init__(self, events: list[Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self.streams: list[OpenAiStream] = []
        self._events = events

    @property
    def responses(self) -> Any:
        outer = self

        class Responses:
            async def create(self, **kwargs: Any) -> Any:
                outer.requests.append(kwargs)
                stream = OpenAiStream(outer._events)
                outer.streams.append(stream)
                return stream

        return Responses()
