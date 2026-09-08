"""The modal screens the slash commands open. Each is a list read with
the keys — the arrows move, Enter chooses, a digit jumps, Escape cancels
— and dismisses with its result, or None (`chooser.py`). The pickers
choose one of a list (session, model, agent); the switchboards (`/mcp`,
`/skill`) stay open, their rows switches; the key prompt takes a
provider, then its key. Which row means "yes" is decided here, at the
edge, never in core."""

from cli.screens.agent import AgentPicker
from cli.screens.key import KeyPrompt
from cli.screens.mcp import McpPicker
from cli.screens.model import ModelPicker
from cli.screens.session import SessionPicker
from cli.screens.skill import SkillPicker

__all__ = [
    "AgentPicker",
    "KeyPrompt",
    "McpPicker",
    "ModelPicker",
    "SessionPicker",
    "SkillPicker",
]
