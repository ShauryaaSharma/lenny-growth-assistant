"""Transcript parsing and chunking.

These matter more than they look: a chunking regression silently degrades every
answer in the product, and does so without raising anything.
"""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    chunk_turns,
    estimate_tokens,
    parse_frontmatter,
    parse_turns,
)

VALID_FRONTMATTER = """---
guest: Adam Fishman
title: How to build a high-performing growth team
youtube_url: https://www.youtube.com/watch?v=abc123
video_id: abc123
publish_date: 2023-04-21
duration_seconds: 3600.0
description: A conversation about growth teams.
keywords:
- growth
- hiring
---

# How to build a high-performing growth team

Adam Fishman (00:00:00):
Onboarding is the only part of your product everyone touches.

Lenny (00:01:30):
That is a great point about onboarding.
"""


class TestFrontmatter:
    def test_parses_all_metadata_fields(self):
        meta, body = parse_frontmatter(VALID_FRONTMATTER, "episodes/adam/transcript.md")
        assert meta.video_id == "abc123"
        assert meta.guest == "Adam Fishman"
        assert meta.publish_date.isoformat() == "2023-04-21"
        assert meta.duration_seconds == 3600.0
        assert meta.keywords == ["growth", "hiring"]
        assert "Onboarding" in body

    def test_content_hash_is_stable_and_content_sensitive(self):
        a, _ = parse_frontmatter(VALID_FRONTMATTER, "p.md")
        b, _ = parse_frontmatter(VALID_FRONTMATTER, "p.md")
        c, _ = parse_frontmatter(VALID_FRONTMATTER.replace("great point", "poor point"), "p.md")
        # Idempotent re-ingestion depends on both halves of this.
        assert a.content_hash == b.content_hash
        assert a.content_hash != c.content_hash

    def test_missing_video_id_falls_back_to_slug(self):
        """Four upstream episodes ship with empty metadata but real transcripts.
        Dropping them would silently shrink the corpus."""
        raw = VALID_FRONTMATTER.replace("video_id: abc123", "video_id: ''").replace(
            "youtube_url: https://www.youtube.com/watch?v=abc123", "youtube_url: ''"
        )
        meta, _ = parse_frontmatter(raw, "episodes/peter-deng/transcript.md")
        assert meta.video_id == "slug:peter-deng"
        assert meta.youtube_url == ""  # no deep link is possible, and we do not fake one

    def test_rejects_missing_frontmatter(self):
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parse_frontmatter("# Just a heading\n", "p.md")

    def test_rejects_unterminated_frontmatter(self):
        with pytest.raises(ValueError, match="unterminated"):
            parse_frontmatter("---\nguest: X\n", "p.md")


class TestTurns:
    def test_extracts_speaker_and_timestamp(self):
        turns = parse_turns(parse_frontmatter(VALID_FRONTMATTER, "p.md")[1])
        assert len(turns) == 2
        assert turns[0].speaker == "Adam Fishman"
        assert turns[0].start_seconds == 0
        assert turns[1].start_seconds == 90

    def test_flags_sponsor_reads(self):
        body = """
Lenny (00:02:00):
This episode is brought to you by Linear, the issue tracker.

Lenny (00:03:00):
Head over to linear.app/lenny to sign up today.

Lenny (00:04:00):
Adam, welcome to the podcast. Let us talk about growth teams.
"""
        turns = parse_turns(body)
        assert turns[0].is_sponsor is True
        assert turns[1].is_sponsor is True, "CTA continuation should stay in the sponsor block"
        assert turns[2].is_sponsor is False, "ordinary conversation must close the block"

    def test_conversation_mentioning_sponsor_words_is_not_flagged(self):
        body = """
Adam Fishman (00:10:00):
We had to decide whether to visit the customer or run a free trial experiment.
"""
        # Secondary markers alone must never open a sponsor block, or real
        # product discussion would be excluded from retrieval.
        assert parse_turns(body)[0].is_sponsor is False


class TestChunking:
    def _turns(self, count: int, words: int = 100):
        body = "".join(
            f"\nSpeaker {i} (00:{i:02d}:00):\n{'word ' * words}\n" for i in range(count)
        )
        return parse_turns(body)

    def test_respects_target_token_budget(self):
        chunks = chunk_turns(self._turns(20), target_tokens=400, overlap_tokens=80)
        assert chunks, "expected chunks"
        # The embedding model truncates past 512 tokens; exceeding it makes the
        # tail of a chunk permanently unretrievable.
        assert max(c.token_count for c in chunks) <= 560

    def test_splits_a_monologue_longer_than_the_budget(self):
        body = "\nGuest (00:00:00):\n" + ". ".join(f"Sentence number {i}" for i in range(400))
        chunks = chunk_turns(parse_turns(body), target_tokens=400, overlap_tokens=80)
        assert len(chunks) > 1, "a long single turn must be split, not emitted whole"
        assert max(c.token_count for c in chunks) <= 560

    def test_chunks_overlap(self):
        chunks = chunk_turns(self._turns(12, words=150), target_tokens=400, overlap_tokens=80)
        assert len(chunks) >= 2
        tail = chunks[0].text.split()[-20:]
        assert any(w in chunks[1].text for w in tail), "expected trailing-turn overlap"

    def test_never_mixes_sponsor_and_content_in_one_chunk(self):
        body = """
Lenny (00:01:00):
This episode is brought to you by Coda, the all-in-one doc.

Adam Fishman (00:02:00):
Growth teams should own the activation metric end to end.
"""
        chunks = chunk_turns(parse_turns(body), target_tokens=400, overlap_tokens=80)
        for c in chunks:
            has_ad = "brought to you by" in c.text.lower()
            assert has_ad == c.is_sponsor, "ad copy leaked into a content chunk"

    def test_ordinals_are_contiguous(self):
        chunks = chunk_turns(self._turns(15), target_tokens=400, overlap_tokens=80)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_token_estimate_is_never_zero():
    # A zero would let an empty chunk slip past the packing budget check.
    assert estimate_tokens("") >= 1
    assert estimate_tokens("hello world") >= 2
