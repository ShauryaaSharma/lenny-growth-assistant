"""Hybrid retrieval and the grounding guard.

These tests write real chunks with real embeddings into Postgres and query them
through the actual SQL in `app.rag.retriever.search` -- the RRF fusion, the
tsvector match, and the HNSW-backed vector search are Postgres-specific and are
only meaningfully tested against a real instance.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from app.db.models import Chunk, Episode
from app.rag.embeddings import embed_passages, embed_query
from app.rag.retriever import search
from tests.conftest import requires_db

pytestmark = requires_db


async def _make_episode(db, **overrides) -> Episode:
    defaults = dict(
        video_id=f"vid-{uuid.uuid4().hex[:8]}",
        guest="Adam Fishman",
        title="How to build a high-performing growth team",
        youtube_url="https://www.youtube.com/watch?v=abc123",
        publish_date=date(2023, 4, 21),
        duration_seconds=3600.0,
        description=None,
        keywords=["growth"],
        source_path="episodes/adam-fishman/transcript.md",
        content_hash=uuid.uuid4().hex,
    )
    defaults.update(overrides)
    ep = Episode(**defaults)
    db.add(ep)
    await db.flush()
    return ep


async def _add_chunk(db, episode: Episode, text: str, ordinal: int, is_sponsor: bool = False):
    vector = None if is_sponsor else embed_passages([text])[0]
    chunk = Chunk(
        episode_id=episode.id,
        ordinal=ordinal,
        speaker="Adam Fishman",
        start_seconds=ordinal * 60,
        end_seconds=ordinal * 60 + 30,
        text=text,
        token_count=len(text.split()),
        is_sponsor=is_sponsor,
        embedding=vector,
    )
    db.add(chunk)
    return chunk


class TestGroundingGuard:
    async def test_relevant_content_clears_the_similarity_floor(self, db):
        ep = await _make_episode(db)
        await _add_chunk(
            db,
            ep,
            "Onboarding is the single most important lever for improving user "
            "retention, because it is the one part of the product every new "
            "user actually experiences.",
            ordinal=0,
        )
        await db.commit()

        result = await search(db, "how do I improve user retention through onboarding")
        assert result.grounded is True
        assert result.chunks[0].guest == "Adam Fishman"
        assert result.best_similarity >= 0.55

    async def test_unrelated_query_does_not_ground(self, db):
        ep = await _make_episode(db)
        await _add_chunk(
            db,
            ep,
            "Onboarding is the single most important lever for improving user "
            "retention in a B2B SaaS product.",
            ordinal=0,
        )
        await db.commit()

        result = await search(db, "what is the best recipe for sourdough bread")
        assert result.grounded is False
        assert result.reason in ("no_results", "below_similarity_threshold")

    async def test_empty_corpus_is_ungrounded_not_an_error(self, db):
        result = await search(db, "anything at all")
        assert result.grounded is False
        assert result.chunks == []

    async def test_empty_query_short_circuits(self, db):
        result = await search(db, "   ")
        assert result.grounded is False
        assert result.reason == "empty_query"


class TestSponsorExclusion:
    async def test_sponsor_chunks_are_never_returned(self, db):
        ep = await _make_episode(db)
        await _add_chunk(
            db,
            ep,
            "This episode is brought to you by Linear, the fastest issue tracker "
            "for modern software teams building growth products.",
            ordinal=0,
            is_sponsor=True,
        )
        await _add_chunk(
            db,
            ep,
            "Growth teams should own the activation metric end to end, from "
            "signup through the first meaningful action in the product.",
            ordinal=1,
        )
        await db.commit()

        result = await search(db, "growth teams and activation metrics")
        assert result.grounded is True
        assert all("brought to you by" not in c.text.lower() for c in result.chunks)
        assert all(not c.text == "" for c in result.chunks)


class TestHybridFusion:
    async def test_exact_keyword_match_is_findable_even_with_weak_semantics(self, db):
        """A rare proper noun a 384-dim embedding might blur is still findable
        via the lexical arm of the fusion."""
        ep = await _make_episode(db, guest="Shreyas Doshi")
        await _add_chunk(
            db,
            ep,
            "Zzyxlorp is the internal codename Shreyas used for the pricing "
            "experiment that shipped last quarter.",
            ordinal=0,
        )
        await db.commit()

        result = await search(db, "Zzyxlorp pricing experiment")
        assert result.grounded is True
        assert any("Zzyxlorp" in c.text for c in result.chunks)

    async def test_citation_carries_a_working_timestamped_url(self, db):
        ep = await _make_episode(
            db, youtube_url="https://www.youtube.com/watch?v=xyz789"
        )
        await _add_chunk(
            db,
            ep,
            "The right time to make your first product manager hire is when "
            "the founder can no longer hold the full roadmap in their head.",
            ordinal=2,
        )
        await db.commit()

        result = await search(db, "when should a founder hire their first PM")
        assert result.grounded is True
        citation = result.chunks[0].as_citation()
        assert citation["url"] == "https://www.youtube.com/watch?v=xyz789&t=120"
        assert citation["guest"] == ep.guest

    async def test_episode_without_youtube_url_cites_by_title_only(self, db):
        """The four upstream episodes with empty metadata must degrade
        gracefully rather than producing a broken or fabricated link."""
        ep = await _make_episode(db, youtube_url="")
        await _add_chunk(
            db,
            ep,
            "Company building requires balancing product velocity against "
            "organizational stability as the team scales past twenty people.",
            ordinal=0,
        )
        await db.commit()

        # Close to the chunk's own wording -- this test is about citation
        # degradation (no youtube_url), not about stress-testing paraphrase
        # matching against a single synthetic chunk with no supporting
        # corpus redundancy behind it.
        result = await search(db, "balancing product velocity against organizational stability")
        assert result.grounded is True
        citation = result.chunks[0].as_citation()
        assert citation["url"] == ""


class TestRetrievalResultFormatting:
    async def test_context_block_numbers_sources_for_citation(self, db):
        ep = await _make_episode(db)
        await _add_chunk(db, ep, "First substantive point about growth loops and virality.", 0)
        await _add_chunk(db, ep, "Second substantive point about growth loops and referrals.", 1)
        await db.commit()

        result = await search(db, "growth loops and virality")
        block = result.as_context_block()
        assert "[1]" in block
        assert ep.guest in block

    async def test_context_block_respects_a_character_budget(self, db):
        ep = await _make_episode(db)
        long_text = "Growth loops compound over time. " * 200
        await _add_chunk(db, ep, long_text, 0)
        await db.commit()

        result = await search(db, "growth loops compounding")
        block = result.as_context_block(max_chars=100)
        assert len(block) <= 150  # header + truncation slack, not the full chunk


async def test_top_k_is_respected(db):
    ep = await _make_episode(db)
    for i in range(10):
        await _add_chunk(
            db, ep, f"Point number {i} about product growth strategy and execution.", i
        )
    await db.commit()

    result = await search(db, "product growth strategy", top_k=3)
    assert len(result.chunks) <= 3
