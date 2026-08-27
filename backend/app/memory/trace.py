"""Execution tracing — SQLite, deliberately a second database.

Every chat turn already produces structured stdout logs (`app.logging`) with a
shared `request_id`. That is sufficient to *read* what happened once, in a
terminal, right after it happened. It is not sufficient to *query* what
happened across many turns, and stdout logs vanish the moment the container
restarts. This module exists for that gap: a durable, queryable, per-span
execution trace — one row per LLM call or tool call, with timing.

**Why SQLite, not another Postgres table.** Trace data is append-only,
high-volume relative to the rest of the schema, and has zero correctness
coupling to the relational data in Postgres (a session can be deleted; its
historical traces are still legitimate operational history, not orphaned
data needing a cascade). That profile is exactly what a lightweight
embedded database is for, and it keeps trace volume from ever affecting the
primary database's size, indexes, or backup story. The file lives at
`TRACE_DB_PATH` (default `/data/traces.db`), on the same persistent volume
already used for the transcript corpus and the embedding model cache.

Writes happen off the event loop via `asyncio.to_thread` -- `sqlite3` is
synchronous, and tracing must never be the thing that makes a chat turn
slower or, worse, fail it. `record_span` swallows its own errors for exactly
that reason: a broken trace write is an operability annoyance, not a reason
to lose a user's answer.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    kind TEXT NOT NULL,           -- 'llm_call' | 'tool_call'
    name TEXT NOT NULL,           -- provider name, or tool name
    started_at TEXT NOT NULL,     -- ISO 8601 UTC
    duration_ms INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    meta TEXT                     -- small JSON blob: free-form extra detail
);
CREATE INDEX IF NOT EXISTS ix_spans_session ON spans(session_id);
CREATE INDEX IF NOT EXISTS ix_spans_request ON spans(request_id);
"""


@dataclass
class Span:
    session_id: str
    request_id: str
    kind: str
    name: str
    duration_ms: int
    ok: bool = True
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict = field(default_factory=dict)


def _db_path() -> Path:
    settings = get_settings()
    return Path(getattr(settings, "trace_db_path", "/data/traces.db"))


@contextmanager
def _connect():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _write_span_sync(span: Span) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO spans (id, session_id, request_id, kind, name, started_at, "
            "duration_ms, ok, prompt_tokens, completion_tokens, meta) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                span.session_id,
                span.request_id,
                span.kind,
                span.name,
                span.duration_ms,
                1 if span.ok else 0,
                span.prompt_tokens,
                span.completion_tokens,
                json.dumps(span.meta, ensure_ascii=False) if span.meta else None,
            ),
        )


async def record_span(span: Span) -> None:
    """Fire-and-forget from the agent loop's perspective: never raises."""
    try:
        await asyncio.to_thread(_write_span_sync, span)
    except Exception as exc:  # noqa: BLE001 - tracing must never break a turn
        log.warning("trace_write_failed", error=str(exc), kind=span.kind, name=span.name)


class Timer:
    """Small helper so call sites read as `with Timer() as t: ...; t.ms`."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = int((time.perf_counter() - self._start) * 1000)


def _read_spans_sync(session_id: str | None, request_id: str | None, limit: int) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT * FROM spans WHERE 1=1"
        params: list[object] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if request_id:
            query += " AND request_id = ?"
            params.append(request_id)
        query += " ORDER BY started_at ASC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


async def read_spans(
    session_id: str | None = None, request_id: str | None = None, limit: int = 200
) -> list[dict]:
    return await asyncio.to_thread(_read_spans_sync, session_id, request_id, limit)
