"""Ingestion against a real database, with a small corpus on disk and the
embedding model replaced by fixed vectors."""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio
from sqlalchemy import func, select

from app.config import get_settings
from app.db import session as db_session
from app.db.models import Chunk, Episode
from app.rag import ingest
from tests.conftest import TEST_DATABASE_URL, requires_db

pytestmark = requires_db

FRONTMATTER = """---
guest: {guest}
title: {guest} episode
video_id: {video_id}
publish_date: 2023-04-21
---

## Transcript
"""


def transcript(root: Path, folder: str, guest: str, video_id: str, body: str) -> None:
    path = root / "episodes" / folder / "transcript.md"
    path.parent.mkdir(parents=True)
    path.write_text(FRONTMATTER.format(guest=guest, video_id=video_id) + body, encoding="utf-8")


@pytest_asyncio.fixture
async def corpus(tmp_path, monkeypatch, db):
    """A corpus directory, with ingestion pointed at it and at the test
    database (whose schema the `db` fixture creates)."""
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("INGEST_EPISODE_LIMIT", "0")
    get_settings.cache_clear()
    await db_session.dispose_engine()
    monkeypatch.setattr(ingest, "ensure_corpus", lambda _url, _path: tmp_path)
    monkeypatch.setattr(ingest, "embed_passages", lambda texts: [[0.1] * 384 for _ in texts])
    yield tmp_path
    await db_session.dispose_engine()


async def test_a_duplicate_video_id_is_skipped_not_overwritten(corpus, db):
    transcript(corpus, "benjamin-lauzier", "Benjamin Lauzier", "same-id",
               "\nBenjamin Lauzier (00:00:01):\nMarketplaces need liquidity first.\n")
    transcript(corpus, "benjamin-mann", "Benjamin Mann", "same-id",
               "\nBenjamin Mann (00:00:01):\nSafety research is product work.\n")

    first = await ingest.run_ingestion()
    assert (first["episodes_ingested"], first["episodes_skipped"]) == (1, 1)
    episode = (await db.execute(select(Episode))).scalar_one()
    assert episode.guest == "Benjamin Lauzier", "the first folder in sorted order is kept"

    # Before the fix, every run overwrote one with the other and re-embedded both.
    second = await ingest.run_ingestion()
    assert (second["episodes_ingested"], second["chunks_written"]) == (0, 0)


async def test_every_transcript_layout_produces_chunks(corpus, db):
    transcript(corpus, "a", "Asha Sharma", "a", "\nAsha Sharma (00:04):\nAgents change orgs.\n")
    transcript(corpus, "b", "Ryan Hoover", "b", "\n[00:00:00] Ryan: Product Hunt began in 2013.\n")
    transcript(corpus, "c", "Adriel Frederick", "c", "\nAdriel Frederick:\nFeed the algorithm.\n")

    summary = await ingest.run_ingestion()

    assert summary["status"] == "ok" and summary["episodes_ingested"] == 3
    assert (await db.execute(select(func.count(Chunk.id)))).scalar_one() == 3
    empty = (await db.execute(
        select(Episode.video_id).where(~Episode.chunks.any()))).scalars().all()
    assert empty == []
