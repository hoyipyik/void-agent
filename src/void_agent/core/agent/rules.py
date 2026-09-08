"""Control-call names and the runtime's default model instructions.
An agent can supplement final-answer guidance through `output(instructions=...)`;
the loop's validation, preemption, and completion rules still apply.
These instructions steer model behavior: changing their wording is a behavior
change and deserves a test."""

from __future__ import annotations

FINAL_ANSWER = "final_answer"
ASK_USER = "ask_user"

# Appended to every agent's system prompt. Load-bearing for correctness, not
# just quality: the loop executes all of one step's tool calls concurrently,
# so a dependent call issued in the same step could only run on fabricated
# arguments — the model must serialize dependent calls across steps itself.
TOOL_SEQUENCING_RULE = (
    "Tool-call sequencing: call multiple tools in one step only when they are fully "
    "independent of each other. When a tool's arguments depend on another tool's "
    "result, call only the prerequisite tool first, and make the dependent call in a "
    "later step once the result is available. Never guess or fabricate an argument "
    "that a previous tool call was supposed to provide."
)

MUST_SUBMIT_RULE = (
    "You answered in plain text, but this task requires a structured submission. "
    "Call the final_answer tool with your answer; that is the only way to finish."
)
MUST_SUBMIT_EMPTY_RULE = (
    "Your last step produced neither message text nor a final_answer call. "
    "Call the final_answer tool with your answer now; that is the only way to finish."
)
# Reasoning-tier models occasionally end a step with no message text at all.
# An empty answer is not an answer; this nudge makes the model actually
# write one.
MUST_SPEAK_RULE = (
    "Your last step produced no message text. Write your final reply to the user "
    "now, as plain message text."
)
INTERNAL_TOOL_ERROR = "Tool execution failed."

FINAL_ANSWER_DESCRIPTION = (
    "Submit the final answer. Calling this tool is the only way to finish the task; "
    "the arguments are the answer itself."
)

# What a gated call reads as when the person never answered: the call did
# not run, and the run is ending — the model meets this text next turn.
HELD_TOOL_OUTPUT = "held for the user's answer; no answer came, so the call did not run"

ASK_USER_DESCRIPTION = (
    "Ask the human. While they are present the answer comes back as this tool's "
    "result and you continue; if no answer comes, the turn ends and the conversation "
    "keeps your question — an ask with no answer in the conversation means the wait "
    "was dropped: ask again, redo the call, or move on, as the situation calls for. "
    "Exhaust your other tools first — retry with what an error tells you, judge what "
    "you can from what you have — and never ask for what you can find out yourself; "
    "ask only when the way forward genuinely depends on the person, and then ask with "
    "this tool rather than stop, guess, or end your turn with a question in text. "
    "Use kind='approval' with a payload naming the exact action and its decisive "
    "fields (amounts, ids, recipients) when you need sign-off before a side effect; "
    "kind='choice' with options when they must pick; kind='input' for a missing fact. "
    "Tools that require approval hold themselves and ask on their own: when a tool "
    "result says the user declined it, or that it was held and did not run, do not "
    "call it again with the same arguments unless the user asks you to."
)
