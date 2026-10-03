"""Knowledge-base ingestion.

Run it directly:      python -m app.rag.ingest
Or in Compose:        docker compose exec backend python -m app.rag.ingest
Then remove episodes no transcript maps to any more:  ... -m app.rag.ingest --prune

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
import difflib
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Episode, IngestionRun
from app.db.session import get_sessionmaker
from app.logging import configure_logging, get_logger
from app.rag.chunking import EpisodeMeta, parse_frontmatter, parse_transcript_file
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


# Transcripts this alike (over their first 3,000 characters) are one episode
# twice, not two episodes. Measured on the corpus: the copies score 0.98-1.00,
# different episodes 0.11-0.33.
SAME_EPISODE_RATIO = 0.9
DUPLICATE, BORROWED = "duplicate", "borrowed"


def resolve_shared_video_ids(files: list[Path]) -> dict[Path, str]:
    """What to do with transcripts whose video id another transcript also has.

    Upstream, 31 video ids each appear in two folders, because one folder's
    title, URL, id and date were copied into the other's frontmatter. Episodes
    are keyed by video id, so the second used to overwrite the first. Here:

    - DUPLICATE: the same transcript twice (7 pairs). Skip the copy.
    - BORROWED: a different episode carrying another's metadata. Ingest it, but
      without that metadata (see `drop_borrowed_metadata`).

    The metadata's owner is the one folder whose guest the title names. When
    that's ambiguous -- no guest named, or both, or "Tomer Cohen" against
    "Tomer Cohen 2.0" -- no folder can be trusted with it, and all are BORROWED.
    """
    groups: dict[str, list[tuple[Path, EpisodeMeta, str]]] = {}
    for path in files:
        try:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"), str(path))
        except (OSError, ValueError):
            continue  # the ingestion loop reports it
        groups.setdefault(meta.video_id, []).append((path, meta, body))

    decisions: dict[Path, str] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        distinct = [group[0]]
        for item in group[1:]:
            if any(_same_episode(item[2], kept[2]) for kept in distinct):
                decisions[item[0]] = DUPLICATE  # the first copy, in sorted order, is kept
            else:
                distinct.append(item)
        if len(distinct) > 1:
            owner = _metadata_owner(distinct)
            decisions.update({path: BORROWED for path, _, _ in distinct if path != owner})
    return decisions


def _same_episode(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a[:3000], b[:3000], autojunk=False).ratio() \
        > SAME_EPISODE_RATIO


def _metadata_owner(group: list[tuple[Path, EpisodeMeta, str]]) -> Path | None:
    title = group[0][1].title.lower()
    named = [(path, meta) for path, meta, _ in group if meta.guest.lower() in title]
    if len(named) != 1:
        return None
    owner_path, owner = named[0]
    if any(owner.guest.lower() in meta.guest.lower()
           for path, meta, _ in group if path != owner_path):
        return None
    return owner_path


def drop_borrowed_metadata(meta: EpisodeMeta, path: Path) -> EpisodeMeta:
    """Keep the episode's own guest and text; drop the id, URL, title, date and
    description it copied from another episode. The same honest fallback as an
    episode whose upstream metadata is empty: cited by guest, with no link."""
    return replace(meta, video_id=f"slug:{path.parent.name}", youtube_url="",
                   title="Untitled episode", publish_date=None, duration_seconds=None,
                   description=None, keywords=[])


def corpus_video_ids(files: list[Path]) -> set[str]:
    """The video ids these transcripts are ingested under: one per episode,
    after copies are skipped and borrowed metadata dropped."""
    shared = resolve_shared_video_ids(files)
    ids: set[str] = set()
    for path in files:
        if shared.get(path) == DUPLICATE:
            continue
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"), str(path))
        except (OSError, ValueError):
            continue
        if shared.get(path) == BORROWED:
            meta = drop_borrowed_metadata(meta, path)
        ids.add(meta.video_id)
    return ids


async def prune_episodes() -> list[str]:
    """Delete episodes that no transcript in the corpus is ingested under any
    more -- removed upstream, or left behind by a change in how shared video
    ids are resolved. Never automatic: a clone that came back partial would
    otherwise empty the knowledge base. Refuses on a subset ingest or an empty
    corpus, for the same reason. Returns the video ids it deleted."""
    settings = get_settings()
    if settings.ingest_episode_limit:
        raise RuntimeError("INGEST_EPISODE_LIMIT is set: pruning would delete every "
                           "episode outside the subset")
    files = discover_transcripts(Path(settings.transcripts_local_path))
    if not files:
        raise RuntimeError(f"no transcripts under {settings.transcripts_local_path}")
    keep = await asyncio.to_thread(corpus_video_ids, files)
    async with get_sessionmaker()() as db:
        stale = list((await db.execute(
            select(Episode.video_id).where(Episode.video_id.not_in(keep))
            .order_by(Episode.video_id))).scalars())
        if stale:
            # Chunks go with their episode (ON DELETE CASCADE).
            await db.execute(delete(Episode).where(Episode.video_id.in_(stale)))
            await db.commit()
    log.info("episodes_pruned", count=len(stale), video_ids=stale)
    return stale


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
    kept_path: dict[str, str] = {}
    error: str | None = None

    try:
        root = await asyncio.to_thread(
            ensure_corpus, settings.transcripts_repo_url, settings.transcripts_local_path
        )
        files = discover_transcripts(root, settings.ingest_episode_limit)
        shared = await asyncio.to_thread(resolve_shared_video_ids, files)
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

            # Folders sharing a video id: see resolve_shared_video_ids.
            if shared.get(path) == DUPLICATE:
                log.warning("duplicate_transcript_skipped", video_id=meta.video_id,
                            path=str(path))
                skipped += 1
                continue
            if shared.get(path) == BORROWED:
                log.warning("borrowed_metadata_dropped", video_id=meta.video_id,
                            path=str(path))
                meta = drop_borrowed_metadata(meta, path)
            # Anything still colliding would overwrite an episode ingested
            # earlier in this run, and re-embed both on every later run.
            if meta.video_id in kept_path:
                log.warning("duplicate_video_id", video_id=meta.video_id,
                            kept=kept_path[meta.video_id], skipped=str(path))
                skipped += 1
                continue
            kept_path[meta.video_id] = str(path)

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

    async def run(force: bool, prune: bool) -> None:
        await run_ingestion(force=force)
        if prune:
            stale = await prune_episodes()
            print(f"pruned {len(stale)} episodes no transcript maps to: "
                  f"{', '.join(stale) or '-'}")

    asyncio.run(run(force="--force" in sys.argv, prune="--prune" in sys.argv))


if __name__ == "__main__":
    main()
