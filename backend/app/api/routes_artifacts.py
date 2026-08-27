"""Artifact endpoints.

Artifacts are always served post-sanitisation — the raw model output was never
stored (see `app.security.sanitize`), so there is no unsafe version to leak.
The client is still responsible for rendering HTML inside a sandboxed iframe;
these two defences are independent by design.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact
from app.db.session import get_db
from app.schemas.api import ArtifactOut, ArtifactSummary

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("", response_model=list[ArtifactSummary])
async def list_artifacts(
    session_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[ArtifactSummary]:
    stmt = select(Artifact).order_by(Artifact.created_at.desc()).limit(limit)
    if session_id is not None:
        stmt = stmt.where(Artifact.session_id == session_id)
    rows = (await db.execute(stmt)).scalars().all()
    return [ArtifactSummary.model_validate(a) for a in rows]


@router.get("/{artifact_id}", response_model=ArtifactOut)
async def get_artifact(
    artifact_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> ArtifactOut:
    artifact = (
        await db.execute(select(Artifact).where(Artifact.id == artifact_id))
    ).scalar_one_or_none()
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "artifact_not_found",
                "message": f"No artifact with id {artifact_id}",
                "hint": "List artifacts for the session with GET /api/artifacts?session_id=...",
            },
        )
    return ArtifactOut.model_validate(artifact)
