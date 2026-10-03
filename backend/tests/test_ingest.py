"""Ingestion: which transcripts become which episodes.

The shared-video-id cases mirror the upstream corpus, where 31 video ids each
appear in two folders. The database tests run real ingestion against Postgres,
with a small corpus on disk and the embedding model replaced by fixed vectors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from app.config import get_settings
from app.db import session as db_session
from app.db.models import Chunk, Episode
from app.rag import ingest
from app.rag.ingest import BORROWED, DUPLICATE, resolve_shared_video_ids
from tests.conftest import TEST_DATABASE_URL, requires_db

FRONTMATTER = """---
guest: {guest}
title: '{title}'
youtube_url: https://www.youtube.com/watch?v={video_id}
video_id: {video_id}
publish_date: 2023-04-21
---

## Transcript
"""

LAUZIER = "\nBenjamin Lauzier (00:00:01):\n" + "Marketplaces need liquidity before anything else. " * 40
MANN = "\nBenjamin Mann (00:00:01):\n" + "Safety research is product work at an AI lab. " * 40
MARKETPLACES = "How marketplaces win | Benjamin Lauzier (Lyft, Thumbtack)"


def transcript(root: Path, folder: str, guest: str, video_id: str, body: str,
               title: str = "") -> Path:
    path = root / "episodes" / folder / "transcript.md"
    path.parent.mkdir(parents=True)
    path.write_text(FRONTMATTER.format(guest=guest, video_id=video_id,
                                       title=title or f"An episode | {guest}") + body,
                    encoding="utf-8")
    return path


# ------------------------------------------------- deciding, without a database

@pytest.mark.unit
def test_borrowed_metadata_belongs_to_the_guest_the_title_names(tmp_path):
    """benjamin-mann carries benjamin-lauzier's title, URL and id upstream."""
    lauzier = transcript(tmp_path, "benjamin-lauzier", "Benjamin Lauzier", "CYw", LAUZIER,
                         MARKETPLACES)
    mann = transcript(tmp_path, "benjamin-mann", "Benjamin Mann", "CYw", MANN, MARKETPLACES)
    assert resolve_shared_video_ids([lauzier, mann]) == {mann: BORROWED}


@pytest.mark.unit
def test_the_owner_is_found_whichever_folder_sorts_first(tmp_path):
    """alexander-embiricos sorts first but carries nilan-peiris's metadata."""
    title = "How to drive word of mouth | Nilan Peiris (CPO of Wise)"
    alex = transcript(tmp_path, "alexander-embiricos", "Alexander Embiricos", "xZi", MANN, title)
    nilan = transcript(tmp_path, "nilan-peiris", "Nilan Peiris", "xZi", LAUZIER, title)
    assert resolve_shared_video_ids([alex, nilan]) == {alex: BORROWED}


@pytest.mark.unit
def test_the_same_transcript_twice_is_a_duplicate(tmp_path):
    first = transcript(tmp_path, "andy-raskin", "Andy Raskin", "dkV", LAUZIER)
    copy = transcript(tmp_path, "andy-raskin_", "Andy Raskin", "dkV", LAUZIER + " Thanks.")
    assert resolve_shared_video_ids([first, copy]) == {copy: DUPLICATE}


@pytest.mark.unit
def test_a_versioned_guest_is_ambiguous_so_neither_keeps_the_metadata(tmp_path):
    """'Tomer Cohen' is named in the title, but so is part of 'Tomer Cohen 2.0':
    the title can't say which episode it belongs to."""
    title = "Why AI is disrupting traditional product management | Tomer Cohen"
    one = transcript(tmp_path, "tomer-cohen", "Tomer Cohen", "R-z", LAUZIER, title)
    two = transcript(tmp_path, "tomer-cohen-20", "Tomer Cohen 2.0", "R-z", MANN, title)
    assert resolve_shared_video_ids([one, two]) == {one: BORROWED, two: BORROWED}


@pytest.mark.unit
def test_unique_video_ids_need_no_decision(tmp_path):
    a = transcript(tmp_path, "a", "A", "id-a", LAUZIER)
    b = transcript(tmp_path, "b", "B", "id-b", MANN)
    assert resolve_shared_video_ids([a, b]) == {}


# ----------------------------------------------------- ingesting, end to end

@pytest_asyncio.fixture
async def corpus(tmp_path, monkeypatch, db):
    """A corpus directory, with ingestion pointed at it and at the test
    database (whose schema the `db` fixture creates)."""
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("INGEST_EPISODE_LIMIT", "0")
    monkeypatch.setenv("TRANSCRIPTS_LOCAL_PATH", str(tmp_path))
    get_settings.cache_clear()
    await db_session.dispose_engine()
    monkeypatch.setattr(ingest, "ensure_corpus", lambda _url, _path: tmp_path)
    monkeypatch.setattr(ingest, "embed_passages", lambda texts: [[0.1] * 384 for _ in texts])
    yield tmp_path
    await db_session.dispose_engine()


@requires_db
async def test_shared_video_ids_no_longer_overwrite_episodes(corpus, db):
    transcript(corpus, "andy-raskin", "Andy Raskin", "dkV", LAUZIER)
    transcript(corpus, "andy-raskin_", "Andy Raskin", "dkV", LAUZIER + " Thanks.")
    transcript(corpus, "benjamin-lauzier", "Benjamin Lauzier", "CYw", LAUZIER, MARKETPLACES)
    transcript(corpus, "benjamin-mann", "Benjamin Mann", "CYw", MANN, MARKETPLACES)

    first = await ingest.run_ingestion()

    assert (first["episodes_ingested"], first["episodes_skipped"]) == (3, 1)
    episodes = {e.guest: e for e in (await db.execute(select(Episode))).scalars()}
    assert set(episodes) == {"Andy Raskin", "Benjamin Lauzier", "Benjamin Mann"}
    lauzier, mann = episodes["Benjamin Lauzier"], episodes["Benjamin Mann"]
    assert (lauzier.video_id, lauzier.title) == ("CYw", MARKETPLACES)
    # Mann's own words, without Lauzier's title, link or date.
    assert (mann.video_id, mann.youtube_url, mann.title, mann.publish_date) == \
        ("slug:benjamin-mann", "", "Untitled episode", None)
    mann_text = (await db.execute(
        select(Chunk.text).where(Chunk.episode_id == mann.id))).scalars().first()
    assert "Safety research" in mann_text

    # Before, each run overwrote one with the other and re-embedded both.
    second = await ingest.run_ingestion()
    assert (second["episodes_ingested"], second["chunks_written"]) == (0, 0)


@requires_db
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


@requires_db
async def test_prune_deletes_episodes_no_transcript_maps_to(corpus, db, monkeypatch):
    """The upgrade path: a row ingested under a shared id that, once resolved,
    no transcript is ingested under any more."""
    transcript(corpus, "tomer-cohen", "Tomer Cohen", "R-z", LAUZIER)
    await ingest.run_ingestion()
    title = "Why AI is disrupting traditional product management | Tomer Cohen"
    for folder, guest, body in (("tomer-cohen", "Tomer Cohen", LAUZIER),
                                ("tomer-cohen-20", "Tomer Cohen 2.0", MANN)):
        path = corpus / "episodes" / folder / "transcript.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(FRONTMATTER.format(guest=guest, video_id="R-z", title=title) + body,
                        encoding="utf-8")
    await ingest.run_ingestion()

    assert await ingest.prune_episodes() == ["R-z"]
    ids = set((await db.execute(select(Episode.video_id))).scalars())
    assert ids == {"slug:tomer-cohen", "slug:tomer-cohen-20"}
    assert await ingest.prune_episodes() == []


@requires_db
async def test_prune_refuses_a_subset_ingest(corpus, monkeypatch):
    monkeypatch.setenv("INGEST_EPISODE_LIMIT", "40")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="INGEST_EPISODE_LIMIT"):
        await ingest.prune_episodes()


@requires_db
async def test_prune_refuses_an_empty_corpus(corpus):
    with pytest.raises(RuntimeError, match="no transcripts"):
        await ingest.prune_episodes()
