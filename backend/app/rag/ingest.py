"""Knowledge-base ingestion.

Run it directly:      python -m app.rag.ingest
Or in Compose:        docker compose exec backend python -m app.rag.ingest

Properties that matter for handoff:

  * **Idempotent.** Each episode's file is hashed; an unchanged episode is
    skipped. Re-running after `git pull` upstream only re-embeds what changed,
    so refreshing the corpus is cheap and safe to schedule.
  * **Auditable.** Every run writes an `ingestion_runs` row with counts, status,
    and any error, so "when was the KB last built and did it work" is a query.
  * **Not vendored.** The corpus is cloned at runtime rather than committed, so
    this repository carries no third-party content.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Episode, IngestionRun
from app.db.session import get_sessionmaker
from app.logging import configure_logging, get_logger
from app.rag.chunking import parse_transcript_file
from app.rag.embeddings import embed_passages

log = get_logger(__name__)

EMBED_BATCH = 128


def ensure_corpus(repo_url: str, local_path: str) -> Path:
    """Clone the transcript repo if absent, otherwise fast-forward it.

    Network failures are non-fatal when a local copy already exists -- a stale
    corpus beats a failed boot.
    """
    path = Path(local_path)
    if (path / "episodes").is_dir():
        try:
            subprocess.run(  # noqa: S603
                ["git", "-C", str(path), "pull", "--ff-only", "--depth", "1"],
                check=True,
                capture_output=True,
                timeout=120,
            )
            log.info("corpus_updated", path=str(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("corpus_update_failed_using_cached", error=str(exc), path=str(path))
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("corpus_cloning", repo=repo_url, path=str(path))
    subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", repo_url, str(path)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return path


def discover_transcripts(root: Path, limit: int = 0) -> list[Path]:
    files = sorted(root.glob("episodes/*/transcript.md"))
    return files[:limit] if limit > 0 else files


async def _upsert_episode(
    db: AsyncSession, meta, chunks, settings
) -> tuple[bool, int]:
    """Insert or refresh one episode. Returns (ingested, chunks_written)."""
    existing = (
        await db.execute(select(Episode).where(Episode.video_id == meta.video_id))
    ).scalar_one_or_none()

    if existing is not None and existing.content_hash == meta.content_hash:
        return False, 0

    if existing is not None:
        # Content changed upstream: drop old chunks so ordinals stay consistent.
        await db.execute(delete(Chunk).where(Chunk.episode_id == existing.id))
        episode = existing
        episode.guest = meta.guest
        episode.title = meta.title
        episode.youtube_url = meta.youtube_url
        episode.publish_date = meta.publish_date
        episode.duration_seconds = meta.duration_seconds
        episode.description = meta.description
        episode.keywords = meta.keywords
        episode.source_path = meta.source_path
        episode.content_hash = meta.content_hash
        episode.ingested_at = datetime.now(UTC)
    else:
        episode = Episode(
            video_id=meta.video_id,
            guest=meta.guest,
            title=meta.title,
            youtube_url=meta.youtube_url,
            publish_date=meta.publish_date,
            duration_seconds=meta.duration_seconds,
            description=meta.description,
            keywords=meta.keywords,
            source_path=meta.source_path,
            content_hash=meta.content_hash,
        )
        db.add(episode)
        await db.flush()

    # Sponsor chunks are persisted for auditability but never embedded -- there
    # is no reason to spend compute making advertisements retrievable.
    embeddable = [c for c in chunks if not c.is_sponsor]
    vectors: list[list[float]] = []
    for i in range(0, len(embeddable), EMBED_BATCH):
        batch = embeddable[i : i + EMBED_BATCH]
        vectors.extend(embed_passages([c.text for c in batch]))
    vector_by_ordinal = {c.ordinal: v for c, v in zip(embeddable, vectors, strict=True)}

    db.add_all(
        [
            Chunk(
                episode_id=episode.id,
                ordinal=c.ordinal,
                speaker=c.speaker,
                start_seconds=c.start_seconds,
                end_seconds=c.end_seconds,
                text=c.text,
                token_count=c.token_count,
                is_sponsor=c.is_sponsor,
                embedding=vector_by_ordinal.get(c.ordinal),
            )
            for c in chunks
        ]
    )
    return True, len(chunks)


async def run_ingestion(force: bool = False) -> dict:
    """Ingest the whole corpus. Returns a summary dict; never raises."""
    settings = get_settings()
    sessionmaker = get_sessionmaker()
    started = time.perf_counter()

    async with sessionmaker() as db:
        run = IngestionRun(source=settings.transcripts_repo_url, status="running")
        db.add(run)
        await db.commit()
        run_id = run.id

    seen = ingested = skipped = chunks_written = 0
    error: str | None = None

    try:
        root = await asyncio.to_thread(
            ensure_corpus, settings.transcripts_repo_url, settings.transcripts_local_path
        )
        files = discover_transcripts(root, settings.ingest_episode_limit)
        log.info("ingestion_started", episodes=len(files), force=force)

        for path in files:
            seen += 1
            try:
                meta, chunks = await asyncio.to_thread(
                    parse_transcript_file,
                    path,
                    settings.chunk_target_tokens,
                    settings.chunk_overlap_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                log.warning("episode_parse_failed", path=str(path), error=str(exc))
                skipped += 1
                continue

            if force:
                meta.content_hash = f"{meta.content_hash}-force-{run_id}"

            # A per-episode transaction keeps a mid-run failure from discarding
            # everything ingested so far.
            async with sessionmaker() as db:
                try:
                    did_ingest, written = await _upsert_episode(db, meta, chunks, settings)
                    await db.commit()
                except Exception as exc:  # noqa: BLE001
                    await db.rollback()
                    log.warning("episode_ingest_failed", video_id=meta.video_id, error=str(exc))
                    skipped += 1
                    continue

            if did_ingest:
                ingested += 1
                chunks_written += written
            else:
                skipped += 1

            if seen % 25 == 0:
                log.info(
                    "ingestion_progress",
                    seen=seen,
                    total=len(files),
                    ingested=ingested,
                    chunks=chunks_written,
                )

        status = "ok"
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        log.error("ingestion_failed", error=error)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    async with sessionmaker() as db:
        row = await db.get(IngestionRun, run_id)
        if row is not None:
            row.status = status
            row.episodes_seen = seen
            row.episodes_ingested = ingested
            row.episodes_skipped = skipped
            row.chunks_written = chunks_written
            row.error = error
            row.finished_at = datetime.now(UTC)
            await db.commit()

    summary = {
        "status": status,
        "episodes_seen": seen,
        "episodes_ingested": ingested,
        "episodes_skipped": skipped,
        "chunks_written": chunks_written,
        "elapsed_ms": elapsed_ms,
        "error": error,
    }
    log.info("ingestion_complete", **summary)
    return summary


async def corpus_is_empty() -> bool:
    async with get_sessionmaker()() as db:
        count = (await db.execute(select(Chunk.id).limit(1))).first()
    return count is None


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    import sys

    asyncio.run(run_ingestion(force="--force" in sys.argv))


if __name__ == "__main__":
    main()
