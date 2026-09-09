"""The widgets: the protocol rendered, and the chrome around it.

The protocol, as a function of a message's
`parts`: text → `Reply`, a dot in the gutter; `dynamic-tool` →
`ToolChip`, a line that folds open; `data-plan` → `PlanCard`;
`data-reflection` → `ReflectionCard`; `data-ask` → `AskCard`, answered
with the keys; any other `data-*` → `DataCard`, folded; `TurnView`
walks the array and mounts them, live and replayed alike. Around the
log: the welcome box, the command menu, the prompt frame and its
composer, the status line, and the panels `/help` and `/status` leave.
The look is `theme.py`: the one Textual theme, every colour a widget
shows quoted from it by name.
"""

from cli.widgets.ask import AskCard
from cli.widgets.attachbar import AttachmentBar
from cli.widgets.cards import PlanCard, ReflectionCard
from cli.widgets.composer import Composer
from cli.widgets.fold import DataCard, Foldable, ToolChip
from cli.widgets.menu import CommandMenu
from cli.widgets.panels import Panel, help_panel
from cli.widgets.prompt import PromptFrame, StatusBar
from cli.widgets.reply import Reply, Said, UserBubble
from cli.widgets.theme import VOID_THEME
from cli.widgets.turn import TurnView
from cli.widgets.welcome import Welcome

__all__ = [
    "VOID_THEME",
    "AskCard",
    "AttachmentBar",
    "CommandMenu",
    "Composer",
    "DataCard",
    "Foldable",
    "Panel",
    "PlanCard",
    "PromptFrame",
    "ReflectionCard",
    "Reply",
    "Said",
    "StatusBar",
    "ToolChip",
    "TurnView",
    "UserBubble",
    "Welcome",
    "help_panel",
]
