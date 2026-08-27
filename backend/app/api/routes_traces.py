"""Trace inspection endpoint.

Directly serves the brief's own observability requirement: "add structured
logs and enough visibility to diagnose model, retrieval, database, and
artifact-rendering failures." Structured stdout logs answer that for one
turn, read right after it happened; this answers it after the fact, for any
turn, without needing container log retention -- see app/memory/trace.py.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.memory import trace

router = APIRouter(prefix="/api/sessions", tags=["traces"])


@router.get("/{session_id}/trace")
async def get_session_trace(
    session_id: uuid.UUID,
    request_id: str | None = Query(default=None, description="Scope to one turn"),
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[dict]:
    """Every recorded span (LLM calls and tool calls) for a session, or for
    one turn within it if `request_id` is given. Returns an empty list for a
    session with no trace history yet -- not an error, since tracing is
    best-effort by design (see trace.record_span)."""
    return await trace.read_spans(session_id=str(session_id), request_id=request_id, limit=limit)
