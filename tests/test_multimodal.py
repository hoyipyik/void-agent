"""Mixed user content survives the agent boundary and provider rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tests.provider_fakes import AnthropicClient, OpenAiClient, chunk, final_message, text_block

from void_agent import (
    Agent,
    Answer,
    AssistantText,
    ImageContent,
    Message,
    ModelStep,
    PdfContent,
    Role,
    ScriptedLlm,
    TextContent,
    TranscriptEntry,
    UserContent,
    call,
    say,
)
from void_agent.providers.anthropic import AnthropicLlm
from void_agent.providers.openai import OpenAiLlm

IMAGE = ImageContent(data=b"image-test", media_type="image/png")
PDF = PdfContent(data=b"%PDF-test", filename="report.pdf")


def mixed_message() -> Message:
    return Message.user("Compare these:", PDF, "against this photo", IMAGE)


# ── the value ─────────────────────────────────────────────────────────────


def test_message_user_wraps_strings_as_text_parts_in_order() -> None:
    assert mixed_message() == Message(
        Role.USER,
        (TextContent("Compare these:"), PDF, TextContent("against this photo"), IMAGE),
    )


def test_message_assistant_is_one_text_part() -> None:
    assert Message.assistant("done") == Message(Role.ASSISTANT, (TextContent("done"),))


def test_assistant_history_is_exactly_one_text_part() -> None:
    with pytest.raises(ValueError, match="one text part"):
        Message(Role.ASSISTANT, (TextContent("previous "), TextContent("answer")))
    with pytest.raises(ValueError, match="one text part"):
        Message(Role.ASSISTANT, (PDF,))


def test_empty_text_and_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="text content must not be empty"):
        TextContent("")
    with pytest.raises(ValueError, match="text content must not be empty"):
        Message.user("")
    with pytest.raises(ValueError, match="message content must not be empty"):
        Message.user()
    for role in (Role.USER, Role.ASSISTANT):
        with pytest.raises(ValueError, match="message content must not be empty"):
            Message(role, ())


def test_empty_binary_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty bytes"):
        PdfContent(data=b"")
    with pytest.raises(ValueError, match="nonempty bytes"):
        ImageContent(data=b"", media_type="image/png")


# ── the transcript ────────────────────────────────────────────────────────


async def test_text_only_history_maps_to_assistant_text_and_user_content() -> None:
    def inspect(transcript: Sequence[TranscriptEntry]) -> ModelStep:
        assert transcript[1] == AssistantText("previous answer")
        assert transcript[2] == UserContent((TextContent("next question"),))
        return say("done")

    agent = Agent(ScriptedLlm([inspect]), "a", "a")
    agent.prompt(lambda _: [Message.assistant("previous answer"), Message.user("next question")])
    assert await agent.run({}) == Answer("done")


async def test_binary_only_input_survives_a_tool_round_trip() -> None:
    def after_tool(transcript: Sequence[TranscriptEntry]) -> ModelStep:
        assert transcript[1] == UserContent((IMAGE, PDF))
        assert len(transcript) == 4  # system, input, tool call, tool return
        return say("done")

    # Even an unknown tool's error result must preserve the original input.
    agent = Agent(ScriptedLlm([call("unknown", {}), after_tool]), "a", "a")
    agent.prompt(lambda _: [Message.user(IMAGE, PDF)])
    assert await agent.run({}) == Answer("done")


# ── the wire ──────────────────────────────────────────────────────────────


async def test_openai_receives_text_pdf_and_image_in_order() -> None:
    fake = OpenAiClient([chunk(content="Compared")])
    agent = Agent(OpenAiLlm("vision-model", client=cast(AsyncOpenAI, fake)), "a", "a")
    agent.prompt(lambda _: [mixed_message()])
    assert await agent.run({}) == Answer("Compared")
    assert fake.requests[0]["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these:"},
            {
                "type": "file",
                "file": {
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi10ZXN0",
                },
            },
            {"type": "text", "text": "against this photo"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2UtdGVzdA=="},
            },
        ],
    }


async def test_anthropic_receives_text_pdf_and_image_in_order() -> None:
    fake = AnthropicClient([], final_message(text_block("Compared")))
    agent = Agent(AnthropicLlm(client=cast(AsyncAnthropic, fake)), "a", "a")
    agent.prompt(lambda _: [mixed_message()])
    assert await agent.run({}) == Answer("Compared")
    assert fake.requests[0]["messages"][0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these:"},
            {
                "type": "document",
                "title": "report.pdf",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi10ZXN0",
                },
            },
            {"type": "text", "text": "against this photo"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aW1hZ2UtdGVzdA==",
                },
            },
        ],
    }
