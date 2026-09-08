"""The person, in two files:

- `attendant` — who answers: the `Attendant` protocol, the ambient
                contextvar (`attended` / `attendant`), and the two
                ready-made attendants — `ScriptedHuman`, the scripted
                seam, and `HumanChannel.channel`, which pairs one with a
                queue of `Question`s, each replied to in place
- `channel`   — how a question travels: `ask_signature` / `ask_words` put
                it to whoever attends the run and hand the answer back to
                the frame that asked; `Unanswered` unwinds when nobody does
"""

from void_agent.core.human.attendant import (
    Attendant,
    HumanChannel,
    Question,
    ScriptedHuman,
    attendant,
    attended,
)
from void_agent.core.human.channel import Unanswered, ask_signature, ask_words

__all__ = [
    "Attendant",
    "HumanChannel",
    "Question",
    "ScriptedHuman",
    "Unanswered",
    "ask_signature",
    "ask_words",
    "attendant",
    "attended",
]
