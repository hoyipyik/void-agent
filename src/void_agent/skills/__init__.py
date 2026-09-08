"""MACHINERY, not a concept: a folder of skills as tools.

A skill is a folder with a SKILL.md — frontmatter naming it and saying
what it is for, then the instructions themselves. Reading it is a tool
call, so only the one-line description stays in the model's context and
the instructions arrive when the model judges them relevant. Twenty
skills cost twenty lines of context, not twenty documents.

A skill carries knowledge, never a capability. Calling one puts text in
the transcript and nothing else happens: no side effect, nothing to sign,
no script run. Capabilities are tools you write in Python, with an
`approval` where they need one — the boundary is what keeps a folder of
Markdown from becoming an execution path.

Nothing here is a runtime concept, which is why there is no
`with_skills()` on `Agent`: a skill has no event and no projection, and
core never opens a path. This is a satellite, like `providers/` and
`mcp/` — it turns something outside into ordinary `Tool`s, and the
runtime never learns the difference.

    for capability in skills_in("~/.void/skills"):
        agent.tool(capability)
"""

from void_agent.skills.shelf import (
    SKILL_FILE,
    SkillFolderError,
    SkillInfo,
    read_skills,
    skills_in,
    tools_for,
)

__all__ = [
    "SKILL_FILE",
    "SkillFolderError",
    "SkillInfo",
    "read_skills",
    "skills_in",
    "tools_for",
]
