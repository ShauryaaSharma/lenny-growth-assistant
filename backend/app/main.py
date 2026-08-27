"""FastAPI application entrypoint.

Resilience posture: the API must start and stay up even when its dependencies
are unhealthy, because an API that refuses to boot cannot tell you *why* it is
unhealthy. A missing API key, an unreachable Ollama, an empty knowledge base, or
a database that is still starting all produce a running server that reports the
problem through `/health/deep` and returns typed errors on the affected routes.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import routes_artifacts, routes_chat, routes_health, routes_sessions
from app.config import get_settings
from app.db.session import check_database, dispose_engine
from app.llm.base import LLMError
from app.llm.registry import close_all
from app.logging import configure_logging, get_logger, request_id_ctx

log = get_logger(__name__)

# Module-level so /health/deep can report on a seed that is still running.
_startup_tasks: set[asyncio.Task] = set()


async def _bootstrap_knowledge_base() -> None:
    """Seed the corpus on first boot so `docker compose up` is genuinely enough.

    Runs in the background: a cold start would otherwise block the API for the
    length of a full ingest. The UI polls /api/config and shows a banner until
    the knowledge base reports ready.
    """
    from app.rag.ingest import corpus_is_empty, run_ingestion

    try:
        if not await corpus_is_empty():
            log.info("knowledge_base_present_skipping_seed")
            return
        log.info("knowledge_base_empty_seeding")
        await run_ingestion()
    except Exception:  # noqa: BLE001 - seeding must never take the API down
        log.exception("knowledge_base_seed_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    log.info(
        "starting",
        app_env=settings.app_env,
        **settings.describe_provider(),
    )

    db_ok, detail = await check_database()
    if not db_ok:
        log.error("database_unreachable_at_startup", detail=detail)

    # Load ONNX weights off the request path.
    from app.rag.embeddings import warm_up

    asyncio.get_running_loop().run_in_executor(None, warm_up)

    if settings.ingest_on_startup and db_ok:
        task = asyncio.create_task(_bootstrap_knowledge_base())
        _startup_tasks.add(task)
        task.add_done_callback(_startup_tasks.discard)

    yield

    for task in list(_startup_tasks):
        task.cancel()
    await close_all()
    await dispose_engine()
    log.info("stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="The Lenny Growth Assistant",
        description=(
            "A grounded conversational assistant over Lenny's Podcast transcripts, "
            "with a Ship 30 for 30 essay skill and sandboxed artifact generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every request with an id and log its outcome and duration."""
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_ctx.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        finally:
            request_id_ctx.reset(token)

        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = rid
        if request.url.path != "/health":  # liveness probes would drown the log
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )
        return response

    def envelope(code: str, message: str, hint: str = "", status_code: int = 500) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "hint": hint,
                    "request_id": request_id_ctx.get(),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException):
        # Routes raise HTTPException with a dict detail carrying code/message/hint.
        if isinstance(exc.detail, dict):
            return envelope(
                exc.detail.get("code", "http_error"),
                exc.detail.get("message", ""),
                exc.detail.get("hint", ""),
                exc.status_code,
            )
        return envelope("http_error", str(exc.detail), "", exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request body failed validation.",
                    "hint": "See `fields` for the offending values.",
                    "request_id": request_id_ctx.get(),
                    "fields": [
                        {"loc": ".".join(str(p) for p in e["loc"]), "msg": e["msg"]}
                        for e in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(LLMError)
    async def llm_error(_request: Request, exc: LLMError):
        return envelope(
            exc.code,
            str(exc),
            "Check GET /health/deep for provider status.",
            exc.http_status,
        )

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        log.exception("unhandled_exception")
        return envelope(
            "internal_error",
            "An unexpected error occurred.",
            "Check the server logs for the matching request_id.",
            500,
        )

    app.include_router(routes_health.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_artifacts.router)
    return app


app = create_app()
