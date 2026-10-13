"""A diagram is the model's own text, never a part.

`draw_diagram` is an ordinary built-in tool, enabled with
`Agent.with_diagram()`, and it draws nothing. The model already writes
Mermaid; what it lacks is the house rules, and those are the call's whole
effect: its result. The diagram follows in the reply as a fenced `mermaid`
block — an ordinary `text` part, rendered by whatever shows the
conversation and never read by the runtime.

So there is no event, no part of its own and nothing to validate. The
rules cost context only when the model is about to draw, the way a skill's
instructions do, and they stay in the transcript as the call's result for
the turns that follow.

The rules are written for a renderer embedded in someone else's page: one
that owns its theme, treats a label as text and follows no link. A more
permissive renderer loses nothing by them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from void_agent.core.tool import Tool

DIAGRAM_DESCRIPTION = (
    "Call this before you draw a chart or a diagram — a bar or line chart, a pie, a "
    "flowchart, a sequence, a state machine, an entity model, a timeline. It takes no "
    "arguments and draws nothing itself: it returns the rules for writing the picture "
    "as Mermaid in your reply, where the user's interface renders it. If those rules "
    "are already in this conversation, follow them without calling again."
)

DIAGRAM_RULES = (
    "Write the chart or diagram in your reply as a fenced code block tagged `mermaid`; "
    "the user's interface renders that block as a picture.\n"
    "\n"
    "- One diagram per block, and nothing inside it but Mermaid: no prose, no Markdown. "
    "Say what it shows in a sentence outside the block.\n"
    "- The first line names the type; choose it by what is being shown. Numbers: "
    "`xychart-beta` with `bar` to rank or compare categories and `line` for a trend "
    "over time, `pie` for the shares of one whole. Structure: `flowchart TD` (or "
    "`LR`), `sequenceDiagram`, `stateDiagram-v2`, `classDiagram`, `erDiagram`, "
    "`mindmap`, `timeline`.\n"
    '- An xychart is `xychart-beta`, then `title "…"`, `x-axis ["A", "B"]`, '
    '`y-axis "Unit" 0 --> 100`, then the series on that one y-axis: `bar [1, 2]` or '
    "`line [1, 2]`. At most one `bar` — bars drawn together hide each other — so "
    'compare several series as lines, and name each one (`line "Paid" [1, 2]`): only '
    "a named series appears in the legend. Scale large numbers (thousands, millions) "
    "and name the unit in the axis title.\n"
    "- At most eight series or slices, and about fifteen nodes; past that, split it "
    "into several diagrams.\n"
    "- One statement per line. Ids are plain ASCII words (`A`, `auth_service`), never "
    "the bare word `end`, which Mermaid reserves; what the reader sees goes in the "
    "label.\n"
    '- In a flowchart, wrap every label in double quotes — `A["Check order (v2)"]`, '
    '`A -->|"yes"| B` — so brackets, colons, slashes and non-ASCII text cannot break '
    "the parse. Never put a double quote inside a label.\n"
    "- No HTML in a label; `<br/>` alone is safe, for a line break.\n"
    "- The interface owns the look: no colors (`classDef`, `style`, `linkStyle`), no "
    "theme, no `%%{init}%%` directive or frontmatter config, no `click`, no images, "
    "icons or math.\n"
    "- To change a diagram, write the whole block again."
)


class DiagramInput(BaseModel):
    """No arguments: the diagram is written in the reply, never passed here."""

    model_config = ConfigDict(extra="forbid")


def diagram_tool() -> Tool:
    """The built-in `draw_diagram` tool. Stateless and silent: its result is
    the rules, and the diagram's home is the reply that follows."""

    async def draw_diagram(_: DiagramInput) -> str:
        return DIAGRAM_RULES

    return Tool(name="draw_diagram", description=DIAGRAM_DESCRIPTION, handler=draw_diagram)
