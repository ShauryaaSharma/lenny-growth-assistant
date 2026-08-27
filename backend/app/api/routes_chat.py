"""The chat endpoint — one request, one complete agent turn.

Persistence ordering matters here and is deliberate: the user's message is
committed *before* the agent runs. If the model then times out, the user's turn
is still in the transcript and the conversation is resumable, rather than the
whole exchange vanishing.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.runtime import run_agent
from app.api.routes_sessions import _to_message_out, load_session
from app.db.models import Artifact, Message
from app.db.session import get_db
from app.llm.base import ChatMessage, LLMError
from app.logging import get_logger
from app.rag.retriever import search
from app.schemas.api import (
    ArtifactOut,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    ToolCallOut,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

TITLE_MAX = 60


async def _load_history(db: AsyncSession, session_id: uuid.UUID) -> list[ChatMessage]:
    """Prior turns for this session only.

    Tool messages are excluded: replaying a previous turn's tool traffic wastes
    context and, on small models, invites them to re-answer the earlier question.
    """
    rows = (
        await db.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.role.in_(("user", "assistant")))
            .order_by(Message.created_at)
        )
    ).scalars().all()
    return [ChatMessage(role=m.role, content=m.content) for m in rows]


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    session = await load_session(db, session_id)
    history = await _load_history(db, session_id)

    user_message = Message(session_id=session_id, role="user", content=payload.message)
    db.add(user_message)

    # First user message names the chat, so the sidebar is readable.
    if not history:
        title = payload.message.strip().replace("\n", " ")
        session.title = title[:TITLE_MAX] + ("..." if len(title) > TITLE_MAX else "")

    await db.commit()

    try:
        result = await run_agent(db, session_id, payload.message, history)
    except LLMError as exc:
        log.error(
            "chat_llm_error", session_id=str(session_id), code=exc.code, error=str(exc)
        )
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "code": exc.code,
                "message": str(exc),
                "hint": (
                    "Check GET /health/deep for provider status. If running locally, "
                    "confirm `ollama serve` is up and the configured model is pulled."
                ),
            },
        ) from exc

    assistant_message = Message(
        session_id=session_id,
        role="assistant",
        content=result.content,
        provider=result.provider,
        model=result.model,
        citations=result.citations or None,
        tool_calls=result.tool_calls or None,
        latency_ms=result.latency_ms,
    )
    db.add(assistant_message)
    await db.flush()

    stored: list[Artifact] = []
    for pending in result.artifacts:
        artifact = Artifact(
            session_id=session_id,
            message_id=assistant_message.id,
            kind=pending.kind,
            title=pending.title,
            content=pending.content,
            sanitizer_report=pending.sanitizer_report,
        )
        db.add(artifact)
        stored.append(artifact)
    await db.flush()
    await db.commit()

    return ChatResponse(
        session_id=session_id,
        message=_to_message_out(assistant_message),
        artifacts=[ArtifactOut.model_validate(a) for a in stored],
        tool_calls=[ToolCallOut(**t) for t in result.tool_calls],
        grounded=result.grounded,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/search", response_model=SearchResponse, tags=["debug"])
async def debug_search(
    payload: SearchRequest, db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """Retrieval without the model.

    Exists so an operator can answer "is this a retrieval problem or a model
    problem?" in one request. That distinction is most of RAG debugging.
    """
    result = await search(db, payload.query, top_k=payload.top_k)
    return SearchResponse(
        query=result.query,
        grounded=result.grounded,
        best_similarity=round(result.best_similarity, 4),
        latency_ms=result.latency_ms,
        results=[
            {
                **c.as_citation(),
                "speaker": c.speaker,
                "excerpt": c.text[:500],
                "text_rank": round(c.text_rank, 5),
            }
            for c in result.chunks
        ],
    )
