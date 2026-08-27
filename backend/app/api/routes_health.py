"""Health and configuration endpoints.

`/health` is the container liveness probe: cheap, no dependencies, always fast.
`/health/deep` is the operator's first stop when something is wrong — it probes
the database, every configured LLM provider, and the knowledge base, and reports
each independently so a failure can be localised in one request rather than by
reading logs.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Episode, IngestionRun
from app.db.session import check_database, get_db
from app.llm.registry import health_all
from app.schemas.api import (
    ConfigResponse,
    DeepHealthResponse,
    HealthResponse,
    KnowledgeBaseStatus,
    ProviderStatus,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness. Deliberately touches nothing external."""
    return HealthResponse(status="ok")


async def _knowledge_base_status(db: AsyncSession) -> KnowledgeBaseStatus:
    episodes = (await db.execute(select(func.count(Episode.id)))).scalar_one()
    chunks = (await db.execute(select(func.count(Chunk.id)))).scalar_one()
    embedded = (
        await db.execute(select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None)))
    ).scalar_one()
    last_run = (
        await db.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1))
    ).scalar_one_or_none()

    return KnowledgeBaseStatus(
        episodes=episodes,
        chunks=chunks,
        embedded_chunks=embedded,
        ready=embedded > 0,
        last_run_status=last_run.status if last_run else None,
        last_run_finished_at=last_run.finished_at if last_run else None,
    )


@router.get("/health/deep", response_model=DeepHealthResponse)
async def health_deep(db: AsyncSession = Depends(get_db)) -> DeepHealthResponse:
    settings = get_settings()

    db_ok, db_detail = await check_database()
    providers = [
        ProviderStatus(
            provider=h.provider,
            model=h.model,
            healthy=h.healthy,
            detail=h.detail,
            latency_ms=h.latency_ms,
            models_available=h.models_available,
        )
        for h in await health_all()
    ]

    try:
        kb = await _knowledge_base_status(db)
    except Exception as exc:  # noqa: BLE001 - a broken DB must not break the probe
        kb = KnowledgeBaseStatus(
            episodes=0, chunks=0, embedded_chunks=0, ready=False, last_run_status=str(exc)[:200]
        )

    active_ok = any(p.healthy and p.provider == settings.llm_provider for p in providers)
    if not db_ok:
        status = "error"
    elif not active_ok or not kb.ready:
        status = "degraded"
    else:
        status = "ok"

    return DeepHealthResponse(
        status=status,
        database={"healthy": db_ok, "detail": db_detail},
        providers=providers,
        knowledge_base=kb,
        config={
            "app_env": settings.app_env,
            "embedding_model": settings.embedding_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "retrieval_min_similarity": settings.retrieval_min_similarity,
            **settings.describe_provider(),
        },
    )


@router.get("/api/config", response_model=ConfigResponse, tags=["config"])
async def get_config(db: AsyncSession = Depends(get_db)) -> ConfigResponse:
    """Everything the UI needs to render the provider badge and KB banner."""
    settings = get_settings()
    described = settings.describe_provider()
    try:
        kb = await _knowledge_base_status(db)
    except Exception:  # noqa: BLE001
        kb = KnowledgeBaseStatus(episodes=0, chunks=0, embedded_chunks=0, ready=False)

    return ConfigResponse(
        provider=str(described["provider"]),
        model=str(described["model"]),
        endpoint=str(described["endpoint"]),
        is_local=bool(described["is_local"]),
        api_key_required=bool(described["api_key_required"]),
        api_key_present=bool(described["api_key_present"]),
        fallback_provider=str(described["fallback_provider"]),
        knowledge_base_ready=kb.ready,
        episodes=kb.episodes,
        chunks=kb.chunks,
    )
