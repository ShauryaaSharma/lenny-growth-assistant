"""The knowledge-base checks.

Each returns a `Check`: `pass`, `fail`, or `warn` (worth knowing, not worth
failing a build). Some guard things the schema already enforces -- the
chunk foreign key, the `vector(384)` column -- because those guarantees can
be bypassed (a bulk load with triggers disabled) or drift (the configured
model changes without a migration), and a check that costs one query is
cheaper than finding out from bad answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rag.ingest import (
    BORROWED,
    DUPLICATE,
    corpus_video_ids,
    discover_transcripts,
    resolve_shared_video_ids,
)

# Lenny's Podcast began in 2019; anything earlier is a parsing error.
DEFAULT_MIN_PUBLISH_DATE = date(2019, 1, 1)
INDEXES = ("ix_chunks_embedding_hnsw", "ix_chunks_tsv")


@dataclass
class Check:
    name: str
    status: str                    # pass | fail | warn
    detail: str
    examples: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


def _result(name: str, bad: int, detail_ok: str, detail_bad: str,
            examples: list[str] | None = None, *, warn_only: bool = False) -> Check:
    if not bad:
        return Check(name, "pass", detail_ok)
    return Check(name, "warn" if warn_only else "fail", detail_bad, (examples or [])[:5])


async def _scalar(db: AsyncSession, sql: str, **params) -> int:
    return int((await db.execute(text(sql), params)).scalar_one() or 0)


async def _column(db: AsyncSession, sql: str, **params) -> list[str]:
    return [str(r) for r in (await db.execute(text(sql), params)).scalars().all()]


# --------------------------------------------------------------- the checks

@dataclass
class Corpus:
    """What ingestion makes of the transcripts on this machine."""
    episode_ids: set[str]                 # the video id each episode is stored under
    shared: dict[Path, str] = field(default_factory=dict)  # DUPLICATE / BORROWED


def load_corpus() -> Corpus | None:
    """The transcripts ingestion would pick up (same discovery, same
    INGEST_EPISODE_LIMIT, same shared-id resolution), or None when the corpus
    isn't on this machine."""
    settings = get_settings()
    root = Path(settings.transcripts_local_path)
    if not root.exists():
        return None
    files = discover_transcripts(root, settings.ingest_episode_limit)
    return Corpus(corpus_video_ids(files), resolve_shared_video_ids(files))


async def episodes_match_corpus(db: AsyncSession, expected: int | None,
                                corpus: Corpus | None = None) -> Check:
    """One episode per transcript, and nothing else.

    Given the corpus, compares the stored video ids with the ones ingestion
    produces from it: a row no transcript maps to is stale (an episode removed
    upstream, or left by an older ingest), and a transcript with no row was
    never ingested. Otherwise compares a count."""
    name = "episode count matches the corpus"
    if expected is None and corpus is not None:
        stored = set(await _column(db, "SELECT video_id FROM episodes"))
        stale = sorted(stored - corpus.episode_ids)
        missing = sorted(corpus.episode_ids - stored)
        problems = []
        if stale:
            problems.append(f"{len(stale)} from no transcript "
                            "(python -m app.rag.ingest --prune removes them)")
        if missing:
            problems.append(f"{len(missing)} transcripts not ingested")
        return _result(name, len(problems), f"{len(stored)} episodes, one per transcript",
                       f"{len(stored)} episodes, expected {len(corpus.episode_ids)}: "
                       + "; ".join(problems), stale + missing)
    if expected is None:
        return Check(name, "warn", "no reference count: pass --expected-episodes or "
                                   "make TRANSCRIPTS_LOCAL_PATH available")
    actual = await _scalar(db, "SELECT count(*) FROM episodes")
    return _result(name, actual != expected, f"{actual} episodes, as expected",
                   f"{actual} episodes, expected {expected}")


def shared_video_ids(corpus: Corpus | None) -> Check:
    """Transcripts that share a video id upstream (see
    `app.rag.ingest.resolve_shared_video_ids`). Ingestion handles them, so this
    is a warning: the fix -- the right metadata -- belongs upstream."""
    name = "no transcripts share a video id"
    if corpus is None:
        return Check(name, "warn", "corpus not available to check")
    copies = sorted(path.parent.name for path, what in corpus.shared.items()
                    if what == DUPLICATE)
    borrowed = sorted(path.parent.name for path, what in corpus.shared.items()
                      if what == BORROWED)
    return _result(name, len(corpus.shared), "every transcript has its own video id",
                   f"{len(copies)} copies skipped; {len(borrowed)} ingested without the "
                   "id, link, title and date they shared", borrowed + copies, warn_only=True)


async def every_episode_has_chunks(db: AsyncSession) -> Check:
    bad = await _column(db, """
        SELECT e.video_id FROM episodes e
        WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.episode_id = e.id)
        ORDER BY e.video_id""")
    return _result("every episode has chunks", len(bad), "no empty episodes",
                   f"{len(bad)} episodes have no chunks", bad)


async def chunk_ordinals_are_contiguous(db: AsyncSession) -> Check:
    """Ordinals 0..n-1 with no gap: a gap means a partial write."""
    bad = await _column(db, """
        SELECT e.video_id FROM episodes e JOIN chunks c ON c.episode_id = e.id
        GROUP BY e.video_id HAVING max(c.ordinal) + 1 <> count(*) OR min(c.ordinal) <> 0
        ORDER BY e.video_id""")
    return _result("chunk ordinals are contiguous", len(bad), "no gaps in any episode",
                   f"{len(bad)} episodes have gaps in their chunk ordinals", bad)


async def no_orphan_chunks(db: AsyncSession) -> Check:
    """Guards the foreign key, which a bulk load with triggers disabled skips."""
    bad = await _column(db, """
        SELECT c.id FROM chunks c
        WHERE NOT EXISTS (SELECT 1 FROM episodes e WHERE e.id = c.episode_id)""")
    return _result("no orphan chunks", len(bad), "every chunk belongs to an episode",
                   f"{len(bad)} chunks point at no episode", bad)


async def no_missing_embeddings(db: AsyncSession) -> Check:
    """Sponsor reads are deliberately not embedded; everything else must be."""
    bad = await _column(db, """
        SELECT c.id FROM chunks c
        WHERE c.embedding IS NULL AND c.is_sponsor = FALSE""")
    return _result("no missing embeddings", len(bad), "every non-sponsor chunk is embedded",
                   f"{len(bad)} non-sponsor chunks have no embedding and can't be found "
                   "by vector search", bad)


async def embedding_dimension(db: AsyncSession) -> Check:
    """The column's declared dimension, the configured model's, and the rows'
    all agree. The column type enforces the rows; this catches the configured
    model changing without a migration."""
    expected = get_settings().embedding_dim
    declared = await _scalar(db, """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'""")
    wrong_rows = await _scalar(db, """
        SELECT count(*) FROM chunks
        WHERE embedding IS NOT NULL AND vector_dims(embedding) <> :dim""", dim=expected)
    problems = []
    if declared != expected:
        problems.append(f"column is vector({declared}) but EMBEDDING_DIM is {expected}")
    if wrong_rows:
        problems.append(f"{wrong_rows} embeddings are not {expected}-dimensional")
    return _result("embedding dimension is correct", len(problems),
                   f"vector({expected}) everywhere", "; ".join(problems))


async def publish_dates_in_range(db: AsyncSession, earliest: date) -> list[Check]:
    """Out of range is a parsing error; missing is worth knowing (a date
    filter can never match the episode) but not a failure."""
    today = date.today()
    out = await _column(db, """
        SELECT video_id || ' (' || publish_date || ')' FROM episodes
        WHERE publish_date < :earliest OR publish_date > :today ORDER BY video_id""",
                        earliest=earliest, today=today)
    missing = await _column(db, """
        SELECT video_id FROM episodes WHERE publish_date IS NULL ORDER BY video_id""")
    return [
        _result("publish dates are in range", len(out), f"all between {earliest} and {today}",
                f"{len(out)} episodes dated outside {earliest}..{today}", out),
        _result("publish dates are present", len(missing), "every episode has a date",
                f"{len(missing)} episodes have no publish date, so date filters skip them",
                missing, warn_only=True),
    ]


async def indexes_are_usable(db: AsyncSession) -> Check:
    """Both retrieval indexes exist and are valid. A failed concurrent build
    leaves an index that exists but is never used."""
    usable = set(await _column(db, """
        SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
        WHERE c.relname = ANY(:names) AND i.indisvalid AND i.indisready""",
                               names=list(INDEXES)))
    missing = [n for n in INDEXES if n not in usable]
    return _result("retrieval indexes are usable", len(missing),
                   "HNSW and full-text indexes valid", f"not usable: {', '.join(missing)}", missing)


async def every_chunk_has_searchable_text(db: AsyncSession) -> Check:
    """A chunk of only stopwords gets an empty tsvector: keyword search can
    never return it."""
    bad = await _column(db, """
        SELECT c.id FROM chunks c
        WHERE c.is_sponsor = FALSE AND length(c.tsv) = 0""")
    return _result("every chunk has searchable text", len(bad),
                   "every non-sponsor chunk has full-text terms",
                   f"{len(bad)} chunks have no full-text terms and can't be found by "
                   "keyword search", bad)


async def sample_is_reachable(db: AsyncSession, sample: int) -> list[Check]:
    """Search for a sample of chunks the way retrieval does, and check each
    comes back: by its own embedding through the vector index, and by a
    phrase from its own text through the full-text index.

    Sequential scans are discouraged for the duration, so the vector lookup
    really goes through HNSW -- an approximate index can, in principle, fail
    to return a stored vector even for an exact-match query."""
    rows = (await db.execute(text("""
        SELECT c.id, c.text FROM chunks c
        WHERE c.is_sponsor = FALSE AND c.embedding IS NOT NULL
        ORDER BY md5(c.id::text) LIMIT :n"""), {"n": sample})).all()
    if not rows:
        return [Check("sample is reachable by vector search", "warn", "no chunks to sample")]

    vector_misses, text_misses = [], []
    await db.execute(text("SET LOCAL enable_seqscan = off"))
    for chunk_id, body in rows:
        found = await _column(db, """
            SELECT c.id FROM chunks c
            WHERE c.embedding IS NOT NULL AND c.is_sponsor = FALSE
            ORDER BY c.embedding <=> (SELECT embedding FROM chunks WHERE id = :id)
            LIMIT 3""", id=chunk_id)
        if str(chunk_id) not in found:
            vector_misses.append(str(chunk_id))

        phrase = " ".join(body.split()[:12])
        hit = await _scalar(db, """
            SELECT count(*) FROM chunks c
            WHERE c.id = :id AND c.tsv @@ plainto_tsquery('english', :phrase)""",
                            id=chunk_id, phrase=phrase)
        if not hit and await _scalar(db, "SELECT length(plainto_tsquery('english', :p)::text)",
                                     p=phrase):
            text_misses.append(str(chunk_id))
    await db.execute(text("RESET enable_seqscan"))

    n = len(rows)
    return [
        _result("sample is reachable by vector search", len(vector_misses),
                f"{n}/{n} sampled chunks found by their own embedding",
                f"{len(vector_misses)}/{n} sampled chunks not found by their own embedding",
                vector_misses),
        _result("sample is reachable by keyword search", len(text_misses),
                f"{n}/{n} sampled chunks found by a phrase from their own text",
                f"{len(text_misses)}/{n} sampled chunks not found by their own text",
                text_misses),
    ]


async def run_all(db: AsyncSession, *, expected_episodes: int | None = None,
                  earliest: date = DEFAULT_MIN_PUBLISH_DATE, sample: int = 20) -> list[Check]:
    corpus = load_corpus()
    return [
        await episodes_match_corpus(db, expected_episodes, corpus),
        shared_video_ids(corpus),
        await every_episode_has_chunks(db),
        await chunk_ordinals_are_contiguous(db),
        await no_orphan_chunks(db),
        await no_missing_embeddings(db),
        await embedding_dimension(db),
        *await publish_dates_in_range(db, earliest),
        await indexes_are_usable(db),
        await every_chunk_has_searchable_text(db),
        *await sample_is_reachable(db, sample),
    ]
