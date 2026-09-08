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


def final_message(*blocks: Any) -> Any:
    return SimpleNamespace(content=list(blocks))


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(call_id: str, name: str, input: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=input)


def chunk(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


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
