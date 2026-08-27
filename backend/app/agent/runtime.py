"""The agent loop.

A tool-calling loop written against `LLMProvider`, so the identical code path
serves both the local Ollama demo and any cloud provider.

Three pieces of deterministic insurance wrap the model's own judgement, because
"the model will remember to search" is not a reliability strategy on a 3B model:

  * **Forced retrieval.** If the model tries to answer a substantive question
    without calling `search_transcripts`, we reject that turn once and make it
    search. Routing accuracy stops being probabilistic.
  * **Ungrounded guard.** If retrieval came back empty, a guard message is
    appended before the final turn so the model cannot quietly fall back on
    parametric knowledge.
  * **Forced artifact creation.** If the user's phrasing strongly indicates a
    document request ("checklist", "one-pager", "template"...) but the model
    answered in chat prose instead of calling `create_artifact`, one nudge
    turn corrects it. This is not hypothetical: observed directly on
    `llama3.2:3b` during manual testing, which searched correctly for "a
    one-page onboarding audit checklist" and then wrote the checklist as a
    plain chat message rather than registering it as a rendered artifact.

Bounded by MAX_ITERATIONS so a model that loops on tool calls cannot run forever.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import (
    FORCE_ARTIFACT_NUDGE,
    FORCE_SEARCH_NUDGE,
    SYSTEM_PROMPT,
    UNGROUNDED_GUARD,
    is_trivial,
    wants_artifact,
)
from app.agent.tools import TOOL_SPECS, PendingArtifact, ToolContext, execute_tool
from app.llm.base import ChatMessage, LLMError
from app.llm.registry import chat_with_fallback
from app.logging import get_logger

log = get_logger(__name__)

MAX_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 20  # sliding window; keeps small models inside their context


@dataclass
class AgentResult:
    content: str
    citations: list[dict] = field(default_factory=list)
    artifacts: list[PendingArtifact] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    grounded: bool = False
    iterations: int = 0


async def run_agent(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_message: str,
    history: list[ChatMessage],
) -> AgentResult:
    """Run one conversational turn to completion."""
    started = time.perf_counter()
    ctx = ToolContext(db=db, session_id=session_id)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        *history[-MAX_HISTORY_MESSAGES:],
        ChatMessage(role="user", content=user_message),
    ]

    needs_grounding = not is_trivial(user_message)
    expects_artifact = wants_artifact(user_message)
    nudged = False
    artifact_nudged = False
    # Separate from ctx.grounded, which is the true "retrieval actually found
    # relevant material" signal reported to the API. Reusing that field to
    # also mean "the ungrounded guard already fired" would make an honest
    # refusal report grounded=True to the caller -- a real bug caught while
    # wiring in the artifact-nudge guard below.
    ungrounded_guard_fired = False
    final_content = ""
    provider = model = ""
    iterations = 0

    for iterations in range(1, MAX_ITERATIONS + 1):
        response = await chat_with_fallback(messages, tools=TOOL_SPECS)
        provider, model = response.provider, response.model

        if response.wants_tools:
            messages.append(
                ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                result = await execute_tool(ctx, call.name, call.arguments)
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            continue

        # The model produced a final answer. Decide whether to accept it.
        if needs_grounding and not ctx.searched and not nudged:
            log.info("forcing_retrieval", session_id=str(session_id))
            nudged = True
            messages.append(ChatMessage(role="assistant", content=response.content))
            messages.append(ChatMessage(role="user", content=FORCE_SEARCH_NUDGE))
            continue

        if ctx.searched and not ctx.grounded and not ungrounded_guard_fired:
            log.info("appending_ungrounded_guard", session_id=str(session_id))
            messages.append(ChatMessage(role="assistant", content=response.content))
            messages.append(ChatMessage(role="user", content=UNGROUNDED_GUARD))
            ungrounded_guard_fired = True  # fires once; next answer is accepted
            continue

        if (
            expects_artifact
            and not ctx.artifacts
            and not artifact_nudged
            # A refused/ungrounded answer should stay refused, not be forced
            # into producing a document the corpus can't actually support.
            and (not ctx.searched or ctx.grounded)
        ):
            log.info("forcing_artifact_creation", session_id=str(session_id))
            artifact_nudged = True
            messages.append(ChatMessage(role="assistant", content=response.content))
            messages.append(ChatMessage(role="user", content=FORCE_ARTIFACT_NUDGE))
            continue

        final_content = response.content
        break

    if not final_content:
        # Ran out of iterations mid tool-loop. Ask once, without tools, for a
        # plain answer rather than returning an empty bubble to the user.
        log.warning("agent_iteration_limit", session_id=str(session_id), iterations=iterations)
        try:
            closing = await chat_with_fallback(
                [*messages, ChatMessage(role="user", content=(
                    "Stop calling tools. Answer now in plain prose using what you have."
                ))],
                tools=None,
            )
            final_content = closing.content
            provider, model = closing.provider, closing.model
        except LLMError:
            final_content = (
                "I wasn't able to complete that request. Please try rephrasing your "
                "question, or ask about a more specific product or growth topic."
            )

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "agent_turn_complete",
        session_id=str(session_id),
        iterations=iterations,
        tools_used=[t["tool"] for t in ctx.tool_log],
        grounded=ctx.grounded,
        artifacts=len(ctx.artifacts),
        latency_ms=latency_ms,
        provider=provider,
        model=model,
    )

    return AgentResult(
        content=final_content,
        citations=ctx.citations,
        artifacts=ctx.artifacts,
        tool_calls=ctx.tool_log,
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        grounded=ctx.grounded,
        iterations=iterations,
    )
