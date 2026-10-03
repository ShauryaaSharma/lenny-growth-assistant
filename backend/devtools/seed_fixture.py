"""Seed a small, fixed knowledge base for API, E2E and load-test runs.

The real ingest clones 303 transcripts and takes hours to embed on a CPU.
CI needs seconds and needs to know exactly what is in the corpus, so this
writes three short episodes -- three guests, three publish years -- with
real embeddings from the configured model. That makes the hybrid search,
the relevance floor and the guest/date filters behave exactly as they do on
the full corpus, just over less text.

Idempotent: episodes are keyed by video_id and skipped if present.

    python -m devtools.seed_fixture
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import date

from sqlalchemy import select

from app.db.models import Chunk, Episode
from app.db.session import dispose_engine, get_sessionmaker
from app.rag.embeddings import embed_passages

FIXTURE = [
    {
        "video_id": "fixture-fishman",
        "guest": "Adam Fishman",
        "title": "How to build a high-performing growth team",
        "publish_date": date(2023, 4, 21),
        "chunks": [
            "Onboarding is the single most important lever for improving user retention, "
            "because it is the one part of the product every new user actually experiences.",
            "A growth team should own a metric, not a feature. Give the team retention as "
            "its north star and let it decide which experiments move it.",
        ],
    },
    {
        "video_id": "fixture-winters",
        "guest": "Casey Winters",
        "title": "Growth loops are the new funnels",
        "publish_date": date(2022, 9, 12),
        "chunks": [
            "Retention is the foundation of every growth loop; without it, acquisition spend "
            "simply leaks out of the bucket.",
            "Growth loops compound because the output of one cycle, like new content or new "
            "users, becomes the input to the next.",
        ],
    },
    {
        "video_id": "fixture-grenier",
        "guest": "Adam Grenier",
        "title": "When to invest in new acquisition channels",
        "publish_date": date(2024, 2, 8),
        "chunks": [
            "Invest in a new acquisition channel only once your existing channels are "
            "saturating; a new channel is a bet on a new audience, not a cheaper version "
            "of the old one.",
            "The best acquisition channels match how your users already discover products, "
            "which is why paid search works for some products and never for others.",
        ],
    },
]


async def seed() -> dict:
    added = 0
    async with get_sessionmaker()() as db:
        for spec in FIXTURE:
            exists = (await db.execute(
                select(Episode.id).where(Episode.video_id == spec["video_id"])
            )).scalar_one_or_none()
            if exists:
                continue
            episode = Episode(
                video_id=spec["video_id"], guest=spec["guest"], title=spec["title"],
                youtube_url=f"https://www.youtube.com/watch?v={spec['video_id']}",
                publish_date=spec["publish_date"], duration_seconds=3600.0,
                description=None, keywords=["fixture"],
                source_path=f"fixture/{spec['video_id']}.md",
                content_hash=hashlib.sha256(spec["video_id"].encode()).hexdigest(),
            )
            db.add(episode)
            await db.flush()
            vectors = embed_passages(spec["chunks"])
            for ordinal, (text, vector) in enumerate(zip(spec["chunks"], vectors, strict=True)):
                db.add(Chunk(
                    episode_id=episode.id, ordinal=ordinal, speaker=spec["guest"],
                    start_seconds=ordinal * 300, end_seconds=ordinal * 300 + 90,
                    text=text, token_count=len(text.split()), is_sponsor=False,
                    embedding=vector,
                ))
            added += 1
        await db.commit()
    await dispose_engine()
    return {"episodes_added": added, "fixture_episodes": len(FIXTURE)}


def main() -> None:
    print(asyncio.run(seed()), flush=True)


if __name__ == "__main__":
    main()
