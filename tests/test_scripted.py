"""The scripted seam: steps may react to the transcript, so a script can
echo something the run produced (an id from a tool result, an answer)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from void_agent import (
    Agent,
    Answer,
    ModelStep,
    ScriptedLlm,
    ToolReturns,
    TranscriptEntry,
    call,
    last_tool_return,
    say,
    tool,
)


class EchoIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


@tool(description="Echoes structured.")
async def echo(input: EchoIn) -> dict[str, str]:
    return {"heard": input.text}


async def test_a_scripted_step_may_read_the_transcript_so_far() -> None:
    def react(transcript: Sequence[TranscriptEntry]) -> ModelStep:
        return say(f"you said {last_tool_return(transcript)['heard']}")

    agent = Agent(ScriptedLlm([call("echo", {"text": "hi"}), react]), "s", "subject").tool(echo)
    assert await agent.run({}) == Answer("you said hi")


def test_last_tool_return_reads_the_newest_return() -> None:
    from void_agent import ToolReturn

    transcript: list[TranscriptEntry] = [
        ToolReturns(returns=(ToolReturn(call_id="1", name="a", content='{"first":1}'),)),
        ToolReturns(returns=(ToolReturn(call_id="2", name="b", content='{"second":2}'),)),
    ]
    assert last_tool_return(transcript) == {"second": 2}
