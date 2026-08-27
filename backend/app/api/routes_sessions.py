"""Session and message endpoints.

Session isolation is structural, not conventional: every read and write is
scoped by `session_id` at the query level, so two chats cannot observe each
other's history even if the client confuses their IDs.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Artifact, Message, Session
from app.db.session import get_db
from app.logging import get_logger
from app.schemas.api import (
    ArtifactSummary,
    CitationOut,
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionSummary,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


async def load_session(db: AsyncSession, session_id: uuid.UUID) -> Session:
    """Fetch a session or raise a 404 with a typed body."""
    session = (
        await db.execute(select(Session).where(Session.id == session_id))
    ).scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "session_not_found",
                "message": f"No session with id {session_id}",
                "hint": "Start a new chat to get a fresh session id.",
            },
        )
    return session


def _to_message_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        provider=m.provider,
        model=m.model,
        citations=[CitationOut(**c) for c in (m.citations or [])],
        latency_ms=m.latency_ms,
        created_at=m.created_at,
    )


@router.post("", response_model=SessionSummary, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate, db: AsyncSession = Depends(get_db)
) -> SessionSummary:
    session = Session(
        title=payload.title or "New chat",
        user_metadata=payload.user_metadata or {},
    )
    db.add(session)
    await db.flush()
    log.info("session_created", session_id=str(session.id))
    return SessionSummary(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.get("", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[SessionSummary]:
    counts = (
        select(Message.session_id, func.count(Message.id).label("n"))
        .group_by(Message.session_id)
        .subquery()
    )
    rows = (
        await db.execute(
            select(Session, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.session_id == Session.id)
            .order_by(Session.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=n,
        )
        for s, n in rows
    ]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionDetail:
    session = (
        await db.execute(
            select(Session).where(Session.id == session_id).options(selectinload(Session.messages))
        )
    ).scalar_one_or_none()
    if session is None:
        await load_session(db, session_id)  # raises the typed 404

    artifacts = (
        await db.execute(
            select(Artifact)
            .where(Artifact.session_id == session_id)
            .order_by(Artifact.created_at.desc())
        )
    ).scalars().all()

    return SessionDetail(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=len(session.messages),
        messages=[_to_message_out(m) for m in session.messages],
        artifacts=[ArtifactSummary.model_validate(a) for a in artifacts],
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    # `from __future__ import annotations` turns the `-> None` return hint into
    # the string "None", which FastAPI resolves into a truthy response model and
    # then rejects as a body on a 204. Declaring it explicitly is the fix.
    response_model=None,
)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Response:
    await load_session(db, session_id)
    # Messages and artifacts cascade at the FK level.
    await db.execute(delete(Session).where(Session.id == session_id))
    log.info("session_deleted", session_id=str(session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
