"""The knowledge-base checks: each passes on good data and fails on the bad
data it exists to catch. Vectors are synthetic, so no embedding model loads."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import text

from app.db.models import Chunk, Episode
from app.validation import checks
from app.validation.__main__ import render
from tests.conftest import requires_db

pytestmark = requires_db

TEXTS = ["Onboarding is the lever for retention.",
         "Growth loops compound when each user brings the next."]


def vector(seed: int) -> list[float]:
    v = [0.0] * 384
    v[seed % 384] = 1.0
    v[(seed * 7 + 3) % 384] = 0.5
    return v


async def seed(db, n: int = 2, published: date | None = date(2023, 4, 1)) -> list[Episode]:
    eps = []
    for e in range(n):
        ep = Episode(video_id=f"vid-{uuid.uuid4().hex[:8]}", guest=f"Guest {e}", title="t",
                     youtube_url="https://www.youtube.com/watch?v=x", publish_date=published,
                     duration_seconds=60.0, source_path="x", content_hash=uuid.uuid4().hex)
        db.add(ep)
        await db.flush()
        for i, body in enumerate(TEXTS):
            db.add(Chunk(episode_id=ep.id, ordinal=i, speaker=ep.guest, start_seconds=i,
                         end_seconds=i + 1, text=body, token_count=len(body.split()),
                         is_sponsor=False, embedding=vector(e * 10 + i)))
        eps.append(ep)
    await db.flush()
    return eps


def by_name(results: list[checks.Check]) -> dict[str, checks.Check]:
    return {c.name: c for c in results}


async def test_a_clean_knowledge_base_passes_every_check(db):
    await seed(db)
    results = await checks.run_all(db, expected_episodes=2)
    assert [c.name for c in results if c.status != "pass"] == []
    assert "12 passed, 0 warnings, 0 failed" in render(results)


async def test_episode_count_must_match_the_corpus(db):
    await seed(db)
    result = await checks.episodes_match_corpus(db, 3)
    assert result.status == "fail" and "expected 3" in result.detail


async def test_without_a_count_the_corpus_is_counted_as_ingestion_would(db, tmp_path,
                                                                     monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / "episodes" / name).mkdir(parents=True)
        (tmp_path / "episodes" / name / "transcript.md").write_text("x")
    monkeypatch.setenv("TRANSCRIPTS_LOCAL_PATH", str(tmp_path))
    await seed(db)

    monkeypatch.setenv("INGEST_EPISODE_LIMIT", "2")
    checks.get_settings.cache_clear()
    assert (await checks.episodes_match_corpus(db, None)).status == "pass"

    monkeypatch.setenv("INGEST_EPISODE_LIMIT", "0")
    checks.get_settings.cache_clear()
    result = await checks.episodes_match_corpus(db, None)
    assert result.status == "fail" and result.detail == "2 episodes, expected 3"


async def test_an_episode_without_chunks_fails(db):
    await seed(db)
    db.add(Episode(video_id="empty", guest="G", title="t", youtube_url="u",
                   publish_date=date(2023, 1, 1), duration_seconds=1.0, source_path="x",
                   content_hash="h"))
    await db.flush()
    result = await checks.every_episode_has_chunks(db)
    assert result.status == "fail" and result.examples == ["empty"]


async def test_a_gap_in_ordinals_fails(db):
    [ep] = await seed(db, 1)
    await db.execute(text("UPDATE chunks SET ordinal = 5 WHERE episode_id = :id AND ordinal = 1"),
                     {"id": ep.id})
    assert (await checks.chunk_ordinals_are_contiguous(db)).status == "fail"


async def test_an_orphan_chunk_fails(db):
    """The foreign key prevents this normally; replica mode is how a bulk
    load gets around it."""
    [ep] = await seed(db, 1)
    await db.execute(text("SET session_replication_role = replica"))
    await db.execute(text("DELETE FROM episodes WHERE id = :id"), {"id": ep.id})
    await db.execute(text("SET session_replication_role = origin"))
    result = await checks.no_orphan_chunks(db)
    assert result.status == "fail" and "2 chunks" in result.detail


async def test_a_missing_embedding_fails_unless_the_chunk_is_a_sponsor_read(db):
    await seed(db, 1)
    await db.execute(text("UPDATE chunks SET embedding = NULL, is_sponsor = TRUE WHERE ordinal = 0"))
    assert (await checks.no_missing_embeddings(db)).status == "pass"
    await db.execute(text("UPDATE chunks SET embedding = NULL WHERE ordinal = 1"))
    assert (await checks.no_missing_embeddings(db)).status == "fail"


async def test_a_configured_dimension_that_differs_from_the_column_fails(db, monkeypatch):
    await seed(db, 1)
    assert (await checks.embedding_dimension(db)).status == "pass"
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    checks.get_settings.cache_clear()
    result = await checks.embedding_dimension(db)
    assert result.status == "fail" and "vector(384) but EMBEDDING_DIM is 768" in result.detail


async def test_publish_dates_out_of_range_fail_and_missing_ones_warn(db):
    await seed(db, 1, published=date(2001, 1, 1))
    await seed(db, 1, published=None)
    in_range, present = await checks.publish_dates_in_range(db, date(2019, 1, 1))
    assert in_range.status == "fail" and "(2001-01-01)" in in_range.examples[0]
    assert present.status == "warn" and present.ok


async def test_a_dropped_index_fails(db):
    await seed(db, 1)
    await db.execute(text("DROP INDEX ix_chunks_tsv"))
    result = await checks.indexes_are_usable(db)
    assert result.status == "fail" and result.examples == ["ix_chunks_tsv"]


async def test_a_chunk_of_only_stopwords_fails(db):
    await seed(db, 1)
    await db.execute(text("UPDATE chunks SET text = 'the and of it' WHERE ordinal = 0"))
    assert (await checks.every_chunk_has_searchable_text(db)).status == "fail"


async def test_sampled_chunks_come_back_through_both_indexes(db):
    await seed(db, 3)
    vector_check, text_check = await checks.sample_is_reachable(db, 20)
    assert vector_check.status == "pass" and vector_check.detail.startswith("6/6")
    assert text_check.status == "pass"
