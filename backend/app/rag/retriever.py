"""Hybrid retrieval over the transcript corpus.

Why hybrid rather than pure vector search:

  * Semantic search alone misses exact-token queries -- product names, metric
    names, a guest's surname -- because a 384-dim embedding blurs rare tokens.
  * Lexical search alone misses paraphrase, which is most of how people ask
    product questions ("how do I know when to hire a PM" vs the transcript's
    "the right time to bring on your first product person").

We run both, then fuse with Reciprocal Rank Fusion. RRF needs no score
calibration between the two systems, which matters because cosine similarity
and ts_rank are not on comparable scales.

**Grounding guard.** RRF scores are rank-derived, so even a query the corpus
cannot answer produces a top hit with a respectable fused score. The guard
therefore keys off raw cosine similarity, which *is* an absolute relevance
signal: if nothing clears `retrieval_min_similarity`, the caller is told the
corpus does not cover the question and the agent must say so rather than
answering from the model's parametric memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.logging import get_logger
from app.rag.embeddings import embed_query

log = get_logger(__name__)

RRF_K = 60  # standard RRF damping constant
CANDIDATE_POOL = 30  # per-arm candidates before fusion


@dataclass
class RetrievedChunk:
    chunk_id: str
    episode_id: str
    text: str
    speaker: str | None
    start_seconds: int | None
    end_seconds: int | None
    guest: str
    title: str
    youtube_url: str
    publish_date: str | None
    vector_similarity: float
    text_rank: float
    rrf_score: float

    @property
    def source_url(self) -> str:
        """Deep-link to the exact moment, when the episode has a YouTube URL."""
        if not self.youtube_url:
            return ""
        if self.start_seconds is None:
            return self.youtube_url
        sep = "&" if "?" in self.youtube_url else "?"
        return f"{self.youtube_url}{sep}t={self.start_seconds}"

    @property
    def timestamp_label(self) -> str:
        if self.start_seconds is None:
            return ""
        h, rem = divmod(self.start_seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def as_citation(self) -> dict:
        """The shape the API and the UI citation panel consume."""
        return {
            "chunk_id": self.chunk_id,
            "episode_title": self.title,
            "guest": self.guest,
            "timestamp": self.timestamp_label,
            "url": self.source_url,
            "publish_date": self.publish_date,
            "score": round(self.rrf_score, 5),
            "similarity": round(self.vector_similarity, 4),
        }


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    grounded: bool = False
    best_similarity: float = 0.0
    latency_ms: int = 0
    reason: str = ""

    def as_context_block(self, max_chars: int = 12000) -> str:
        """Render retrieved chunks as numbered evidence for the model prompt.

        Numbering is what lets the model cite: it is instructed to reference
        [1], [2]... and we map those back to real sources on the way out.
        """
        parts: list[str] = []
        used = 0
        for i, c in enumerate(self.chunks, start=1):
            stamp = f" at {c.timestamp_label}" if c.timestamp_label else ""
            header = f"[{i}] {c.guest} - {c.title}{stamp}"
            block = f"{header}\n{c.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts)


_SEARCH_SQL = text(
    """
    WITH vector_arm AS (
        SELECT c.id,
               1 - (c.embedding <=> CAST(:qvec AS vector)) AS similarity,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> CAST(:qvec AS vector)) AS rank
        FROM chunks c
        WHERE c.embedding IS NOT NULL AND c.is_sponsor = FALSE
        ORDER BY c.embedding <=> CAST(:qvec AS vector)
        LIMIT :pool
    ),
    text_arm AS (
        SELECT c.id,
               ts_rank(c.tsv, plainto_tsquery('english', :qtext)) AS rank_score,
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank(c.tsv, plainto_tsquery('english', :qtext)) DESC
               ) AS rank
        FROM chunks c
        WHERE c.tsv @@ plainto_tsquery('english', :qtext) AND c.is_sponsor = FALSE
        LIMIT :pool
    ),
    fused AS (
        SELECT COALESCE(v.id, t.id) AS id,
               COALESCE(v.similarity, 0.0) AS similarity,
               COALESCE(t.rank_score, 0.0) AS text_rank,
               COALESCE(1.0 / (:rrf_k + v.rank), 0.0)
             + COALESCE(1.0 / (:rrf_k + t.rank), 0.0) AS rrf_score
        FROM vector_arm v
        FULL OUTER JOIN text_arm t ON v.id = t.id
    )
    SELECT f.id AS chunk_id, f.similarity, f.text_rank, f.rrf_score,
           c.text, c.speaker, c.start_seconds, c.end_seconds,
           e.id AS episode_id, e.guest, e.title, e.youtube_url, e.publish_date
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN episodes e ON e.id = c.episode_id
    ORDER BY f.rrf_score DESC
    LIMIT :top_k
    """
)


async def search(
    db: AsyncSession,
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> RetrievalResult:
    """Hybrid search. Never raises on an empty corpus -- returns ungrounded."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    min_similarity = (
        min_similarity if min_similarity is not None else settings.retrieval_min_similarity
    )
    started = time.perf_counter()

    query = (query or "").strip()
    if not query:
        return RetrievalResult(query=query, grounded=False, reason="empty_query")

    qvec = embed_query(query)
    rows = (
        await db.execute(
            _SEARCH_SQL,
            {
                "qvec": str(qvec),
                "qtext": query,
                "pool": CANDIDATE_POOL,
                "rrf_k": RRF_K,
                "top_k": top_k,
            },
        )
    ).mappings().all()

    chunks = [
        RetrievedChunk(
            chunk_id=str(r["chunk_id"]),
            episode_id=str(r["episode_id"]),
            text=r["text"],
            speaker=r["speaker"],
            start_seconds=r["start_seconds"],
            end_seconds=r["end_seconds"],
            guest=r["guest"],
            title=r["title"],
            youtube_url=r["youtube_url"] or "",
            publish_date=r["publish_date"].isoformat() if r["publish_date"] else None,
            vector_similarity=float(r["similarity"] or 0.0),
            text_rank=float(r["text_rank"] or 0.0),
            rrf_score=float(r["rrf_score"] or 0.0),
        )
        for r in rows
    ]

    best = max((c.vector_similarity for c in chunks), default=0.0)
    grounded = bool(chunks) and best >= min_similarity
    reason = "" if grounded else ("no_results" if not chunks else "below_similarity_threshold")
    latency_ms = int((time.perf_counter() - started) * 1000)

    log.info(
        "retrieval_complete",
        query_chars=len(query),
        hits=len(chunks),
        best_similarity=round(best, 4),
        grounded=grounded,
        latency_ms=latency_ms,
    )

    return RetrievalResult(
        chunks=chunks,
        query=query,
        grounded=grounded,
        best_similarity=best,
        latency_ms=latency_ms,
        reason=reason,
    )
