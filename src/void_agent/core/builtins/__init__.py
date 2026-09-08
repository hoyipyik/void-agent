"""Built-in capabilities: tools the framework ships, chain-enabled on the
agent (`with_plan`, `with_reflection`) — never constructed or injected by
the app.

Not mechanisms — instances. `core/tool/` defines what a tool IS; this
package holds the tools void_agent provides out of the box, all of them
stateless projections (losing their output never causes wrong behavior):

- `plan`       — update_plan: the model's live task list
- `reflection` — reflect: the model's structured self-assessment

Only the input/item types are public — the tools themselves live inside
the agent.
"""

from void_agent.core.builtins.plan import PlanItem, PlanStatus, PlanUpdate
from void_agent.core.builtins.reflection import ReflectInput

__all__ = [
    "PlanItem",
    "PlanStatus",
    "PlanUpdate",
    "ReflectInput",
]
