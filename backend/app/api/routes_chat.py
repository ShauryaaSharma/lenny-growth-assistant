"""The chat endpoints — one request, one complete agent turn, either as one
JSON response (`/chat`) or reported as it runs (`/chat/stream`).

Persistence ordering matters here and is deliberate: the user's message is
committed *before* the agent runs. If the model then times out, the user's turn
is still in the transcript and the conversation is resumable, rather than the
whole exchange vanishing.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.runtime import AgentResult, run_agent
from app.api.routes_sessions import _to_message_out, load_session
from app.db.models import Artifact, Message
from app.db.session import get_db, get_session_factory
from app.llm.base import ChatMessage, LLMError
from app.logging import get_logger, request_id_ctx
from app.rag.retriever import SearchFilters, search
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


LLM_ERROR_HINT = (
    "Check GET /health/deep for provider status. If running locally, "
    "confirm `ollama serve` is up and the configured model is pulled."
)

# A comment line every this-many seconds while the model is thinking. Proxies
# and load balancers close connections that sit silent for ~30-60s, and one
# CPU-bound model call alone can take longer than that.
KEEPALIVE_SECONDS = 15.0

# Streamed turns run as tasks that outlive their request (see chat_stream).
# Holding a reference keeps them from being garbage-collected mid-turn.
_running_turns: set[asyncio.Task] = set()


async def _begin_turn(db: AsyncSession, session_id: uuid.UUID, message: str) -> list[ChatMessage]:
    """Record the user's turn and return the history before it. Committed
    before the agent runs, so a model failure leaves a resumable chat."""
    session = await load_session(db, session_id)
    history = await _load_history(db, session_id)

    db.add(Message(session_id=session_id, role="user", content=message))

    # First user message names the chat, so the sidebar is readable.
    if not history:
        title = message.strip().replace("\n", " ")
        session.title = title[:TITLE_MAX] + ("..." if len(title) > TITLE_MAX else "")

    await db.commit()
    return history


async def _finish_turn(db: AsyncSession, session_id: uuid.UUID, result: AgentResult) -> ChatResponse:
    """Persist the assistant's turn and its artifacts."""
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


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    history = await _begin_turn(db, session_id, payload.message)

    try:
        result = await run_agent(db, session_id, payload.message, history)
    except LLMError as exc:
        log.error(
            "chat_llm_error", session_id=str(session_id), code=exc.code, error=str(exc)
        )
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc), "hint": LLM_ERROR_HINT},
        ) from exc

    return await _finish_turn(db, session_id, result)


@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> StreamingResponse:
    """The same turn as `/chat`, reported as it happens over Server-Sent Events.

    Events, each `event: <type>` with a JSON `data:` line:

      thinking    a model call is starting (`iteration`)
      tool_start  a tool is about to run (`tool`, `detail`: the query or title)
      tool_end    it finished (`tool`, `ok`, `latency_ms`, `summary`)
      guard       a grounding guard intervened (`guard`)
      done        the turn is saved (`response`: exactly what `/chat` returns)
      error       it failed (`error`: the usual code / message / hint envelope)

    The answer itself arrives only in `done`, never as draft tokens -- the
    agent's guards can still reject a draft (see app.agent.runtime).

    Validation and a missing session fail as ordinary JSON errors before the
    stream opens. After that the turn runs as its own task with its own
    database session: if the client disconnects, the answer is still saved
    and appears when the chat is reopened.
    """
    history = await _begin_turn(db, session_id, payload.message)
    events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def run_turn() -> None:
        try:
            async with session_factory() as turn_db:
                result = await run_agent(turn_db, session_id, payload.message, history,
                                         on_event=events.put)
                response = await _finish_turn(turn_db, session_id, result)
            await events.put({"type": "done", "response": response.model_dump(mode="json")})
        except LLMError as exc:
            log.error("chat_llm_error", session_id=str(session_id), code=exc.code, error=str(exc))
            await events.put(_error_event(exc.code, str(exc), LLM_ERROR_HINT, exc.http_status))
        except Exception:  # noqa: BLE001 - the stream must end with an event, not hang
            log.exception("chat_stream_failed", session_id=str(session_id))
            await events.put(_error_event(
                "internal_error", "An unexpected error occurred.",
                "Check the server logs for the matching request_id.", 500))
        finally:
            await events.put(None)

    task = asyncio.create_task(run_turn())
    _running_turns.add(task)
    task.add_done_callback(_running_turns.discard)

    async def sse() -> AsyncIterator[str]:
        while True:
            try:
                event = await asyncio.wait_for(events.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if event is None:
                return
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        # no-transform and X-Accel-Buffering stop proxies (nginx in
        # particular) from buffering the stream into one late response.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def _error_event(code: str, message: str, hint: str, status_code: int) -> dict[str, Any]:
    return {"type": "error", "status": status_code,
            "error": {"code": code, "message": message, "hint": hint,
                      "request_id": request_id_ctx.get()}}


@router.post("/search", response_model=SearchResponse, tags=["debug"])
async def debug_search(
    payload: SearchRequest, db: AsyncSession = Depends(get_db)
) -> SearchResponse:
    """Retrieval without the model.

    Exists so an operator can answer "is this a retrieval problem or a model
    problem?" in one request. That distinction is most of RAG debugging.
    """
    filters = SearchFilters(guest=(payload.guest or "").strip() or None,
                            since=payload.since, until=payload.until)
    result = await search(db, payload.query, top_k=payload.top_k, filters=filters)
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
