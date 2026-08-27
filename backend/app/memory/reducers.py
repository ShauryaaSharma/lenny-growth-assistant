"""Reducers: pure functions that merge new state into accumulated state.

The term is borrowed from LangGraph/Redux-style state management: instead of
scattering "append this to the running list" logic inline wherever a turn
happens to need it, the merge rule is a single named, pure, independently
testable function. This module has one job -- assembling the message list an
agent turn actually sees -- done explicitly rather than ad hoc.

**Where conversation history actually lives.** This module does not persist
anything. Durable conversation history (the requirement this was written to
satisfy) already exists and is untouched: every message is written to
Postgres in `db/models.Message`, scoped by `session_id`, and reloaded by
`api/routes_chat._load_history` on every turn. That is the system's episodic
memory. Duplicating it into a second database here would mean two sources of
truth for the same fact with no way to guarantee they agree -- a correctness
risk for no real benefit, so this module is deliberately about *shaping* state
for one turn, not storing it.

**Where "another database" earns its keep instead** is `trace.py` in this
same package: execution traces are append-only, ephemeral-by-nature debug
data with no correctness requirement to stay consistent with anything else,
which is exactly the profile that justifies a separate, lightweight SQLite
store instead of another Postgres table.
"""

from __future__ import annotations

from app.llm.base import ChatMessage


def reduce_history(
    history: list[ChatMessage],
    max_messages: int,
) -> list[ChatMessage]:
    """The windowing reducer: bound history to the last `max_messages` turns.

    A sliding window, not summarisation (see docs/PRD.md's scope table for why
    summarisation was cut) -- but naming it as its own reducer means the
    windowing policy is one function to change, and one function to test,
    instead of an inline slice at the call site.
    """
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def reduce_turn(
    history: list[ChatMessage],
    user_message: str,
    assistant_reply: str | None,
) -> list[ChatMessage]:
    """Merge one completed turn into history.

    Appends the user message, then the assistant's reply if there is one yet
    (there won't be, mid-turn, before the model has answered). This is the
    single place "what does the next turn see" is decided -- used both by the
    live agent loop (via `build_agent_messages` below) and by the agent-eval
    harness (`app/evals/run_agent_eval.py`) to thread multi-turn scenarios,
    so both paths accumulate conversation state identically rather than each
    reimplementing it slightly differently.
    """
    merged = [*history, ChatMessage(role="user", content=user_message)]
    if assistant_reply is not None:
        merged.append(ChatMessage(role="assistant", content=assistant_reply))
    return merged


def build_agent_messages(
    system_prompt: str,
    history: list[ChatMessage],
    user_message: str,
    max_history_messages: int,
) -> list[ChatMessage]:
    """The reducer chain that produces exactly what one agent turn sees:
    system prompt, then windowed history, then the new user message. Kept as
    one function so `agent/runtime.py` has a single, named call rather than
    assembling the list inline."""
    windowed = reduce_history(history, max_history_messages)
    return [
        ChatMessage(role="system", content=system_prompt),
        *windowed,
        ChatMessage(role="user", content=user_message),
    ]
