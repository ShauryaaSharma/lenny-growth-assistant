"""Guest and date filters on search.

The integration tests write real chunks with real embeddings and run the
actual SQL. The one that matters most is
`test_filter_applies_before_the_candidate_pool_is_cut`: filtering after the
per-arm pool is cut would silently drop the guest asked about whenever other
guests talk about the topic more.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

import app.agent.tools as tools_module
from app.agent.runtime import _describe_call, _summarize_result
from app.agent.tools import ToolContext, execute_tool, match_guest_names
from app.db.models import Chunk, Episode
from app.rag.embeddings import embed_passages
from app.rag.retriever import (
    CANDIDATE_POOL,
    RetrievalResult,
    SearchFilters,
    parse_date_bound,
    search,
)
from tests.conftest import requires_db
from tests.test_api import client_for, sessionmaker  # noqa: F401 - fixture

# ------------------------------------------------------------ date parsing


@pytest.mark.parametrize("value, end, expected", [
    ("2023", False, date(2023, 1, 1)),
    ("2023", True, date(2023, 12, 31)),
    ("2024-02", True, date(2024, 2, 29)),       # leap year
    ("2023-02", True, date(2023, 2, 28)),
    ("2023-05", False, date(2023, 5, 1)),
    ("2023-05-17", True, date(2023, 5, 17)),
    (" 2023-5-7 ", False, date(2023, 5, 7)),
    (date(2020, 1, 2), True, date(2020, 1, 2)),
    (None, False, None),
    ("", True, None),
])
@pytest.mark.unit
def test_parse_date_bound(value, end, expected):
    assert parse_date_bound(value, end=end) == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", ["last year", "2023/05", "05-2023", "2023-13", "2023-02-30"])
def test_parse_date_bound_rejects(value):
    with pytest.raises(ValueError):
        parse_date_bound(value)


@pytest.mark.unit
def test_filters_reject_a_reversed_range():
    with pytest.raises(ValueError, match="after"):
        SearchFilters.parse(since="2024", until="2023")


@pytest.mark.parametrize("kwargs, text", [
    ({"guest": "Casey Winters"}, "Casey Winters"),
    ({"since": "2023"}, "since 2023-01-01"),
    ({"until": "2023"}, "until 2023-12-31"),
    ({"guest": " Casey ", "since": "2023", "until": "2023"}, "Casey, 2023-01-01 to 2023-12-31"),
])
@pytest.mark.unit
def test_describe(kwargs, text):
    assert SearchFilters.parse(**kwargs).describe() == text


@pytest.mark.unit
def test_blank_guest_is_no_filter():
    assert SearchFilters.parse(guest="   ").active is False


@pytest.mark.unit
def test_like_wildcards_in_a_guest_name_are_literal():
    _, params = SearchFilters(guest="100%_real").where()
    assert params["guest"] == r"%100\%\_real%"


# ------------------------------------------------------------- the tool

@pytest.fixture
def captured_search(monkeypatch):
    calls = []

    async def fake_search(db, query, top_k=None, min_similarity=None, filters=None):
        calls.append(filters)
        if filters and filters.guest == "Nobody":
            return RetrievalResult(query=query, reason="no_matching_episodes")
        return RetrievalResult(query=query, reason="below_similarity_threshold")

    monkeypatch.setattr(tools_module, "search", fake_search)
    return calls


async def run_tool(args):
    return await execute_tool(ToolContext(db=None, session_id=uuid.uuid4()),
                              "search_transcripts", args)


@pytest.mark.unit
async def test_tool_passes_filters_through(captured_search):
    await run_tool({"query": "retention", "guest": "Casey Winters", "since": "2023"})
    assert captured_search == [SearchFilters(guest="Casey Winters", since=date(2023, 1, 1))]


@pytest.mark.unit
async def test_tool_without_filters_searches_unfiltered(captured_search):
    await run_tool({"query": "retention"})
    assert captured_search[0].active is False


@pytest.mark.unit
async def test_tool_reports_a_bad_date_to_the_model(captured_search):
    result = await run_tool({"query": "retention", "since": "last spring"})
    assert "YYYY" in result["error"]
    assert captured_search == []


@pytest.mark.unit
async def test_tool_says_when_no_episode_matches_rather_than_no_topic(captured_search):
    result = await run_tool({"query": "retention", "guest": "Nobody"})
    assert result["grounded"] is False
    assert "No episode in the transcripts matches Nobody" in result["instruction"]
    assert "do not cover this topic" not in result["instruction"]


# ------------------------------------------- guests named in the question

GUESTS = ["Adam Fishman", "Adam Grenier", "Casey Winters", "Aishwarya Reganti + Kiriti Badam"]


@pytest.mark.parametrize("message, named", [
    ("What did Adam Grenier say about acquisition channels?", ["Adam Grenier"]),
    ("what did casey winters say about retention", ["Casey Winters"]),
    ("What does Kiriti Badam think about AI products?", ["Kiriti Badam"]),
    ("Compare Casey Winters and Adam Fishman on growth", ["Adam Fishman", "Casey Winters"]),
    ("What did Adam say about growth teams?", []),           # first name: two Adams
    ("What did Elon Musk say about growth?", []),            # not in the corpus
    ("Casey Wintersmith's take?", []),                        # not a whole-name match
    ("How do I improve retention?", []),
])
@pytest.mark.unit
def test_match_guest_names(message, named):
    assert match_guest_names(message, GUESTS) == named


@pytest.mark.slow
@requires_db
async def test_tool_filters_by_a_guest_the_user_named_when_the_model_did_not(db):
    """What llama3.2:3b actually does: the name goes into the query text."""
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await chunks(db, await episode(db, "Adam Fishman"), [RETENTION])
    await db.commit()

    ctx = ToolContext(db=db, session_id=uuid.uuid4(),
                      user_message="What did Casey Winters say about user retention?")
    result = await execute_tool(ctx, "search_transcripts",
                                {"query": "Casey Winters onboarding user retention"})

    assert result["filters"] == "Casey Winters"
    assert result["grounded"] is True
    assert {r["guest"] for r in result["results"]} == {"Casey Winters"}


@pytest.mark.slow
@requires_db
async def test_nothing_relevant_from_the_guest_says_so_in_those_words(db):
    """Not 'the podcast never covers this' -- only that this guest didn't."""
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await db.commit()

    ctx = ToolContext(db=db, session_id=uuid.uuid4(), user_message="q")
    result = await execute_tool(ctx, "search_transcripts",
                                {"query": "best sourdough starter recipe", "guest": "Casey"})

    assert (result["grounded"], result["filters"]) == (False, "Casey")
    assert "Episodes matching Casey exist" in result["instruction"]
    assert "no_matching_episodes" not in result
    assert _summarize_result("search_transcripts", result) == "nothing relevant found (Casey)"


@pytest.mark.slow
@requires_db
async def test_two_named_guests_are_not_filtered_to_one(db):
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await chunks(db, await episode(db, "Adam Fishman"), [RETENTION])
    await db.commit()

    ctx = ToolContext(db=db, session_id=uuid.uuid4(),
                      user_message="Compare Casey Winters and Adam Fishman on user retention")
    result = await execute_tool(ctx, "search_transcripts", {"query": "user retention"})

    assert "filters" not in result
    assert {r["guest"] for r in result["results"]} == {"Casey Winters", "Adam Fishman"}


@pytest.mark.slow
@requires_db
async def test_the_models_own_guest_argument_wins(db):
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await chunks(db, await episode(db, "Adam Fishman"), [RETENTION])
    await db.commit()

    ctx = ToolContext(db=db, session_id=uuid.uuid4(),
                      user_message="What did Casey Winters say about user retention?")
    result = await execute_tool(ctx, "search_transcripts",
                                {"query": "user retention", "guest": "Adam Fishman"})

    assert {r["guest"] for r in result["results"]} == {"Adam Fishman"}


@pytest.mark.unit
def test_progress_feed_shows_filters():
    args = {"query": "retention", "guest": "Casey Winters", "since": "2023"}
    assert _describe_call("search_transcripts", args) == "retention (Casey Winters, since 2023-01-01)"
    assert _describe_call("search_transcripts", {"query": "q", "since": "bad"}) == "q"
    assert _summarize_result("search_transcripts", {
        "grounded": False, "no_matching_episodes": True, "filters": "Nobody", "results": [],
    }) == "no episode matches Nobody"
    assert _summarize_result("search_transcripts", {
        "grounded": True, "filters": "Casey Winters",
        "results": [{"episode": "a"}, {"episode": "a"}],
    }) == "2 passages from 1 episode (Casey Winters)"


# ----------------------------------------------------- against real SQL

async def episode(db, guest: str, published: date | None = date(2023, 4, 1)) -> Episode:
    ep = Episode(video_id=f"vid-{uuid.uuid4().hex[:8]}", guest=guest, title=f"{guest} episode",
                 youtube_url="https://www.youtube.com/watch?v=x", publish_date=published,
                 duration_seconds=3600.0, source_path="x", content_hash=uuid.uuid4().hex)
    db.add(ep)
    await db.flush()
    return ep


async def chunks(db, ep: Episode, texts: list[str]) -> None:
    for i, (text, vector) in enumerate(zip(texts, embed_passages(texts), strict=True)):
        db.add(Chunk(episode_id=ep.id, ordinal=i, speaker=ep.guest, start_seconds=i * 60,
                     end_seconds=i * 60 + 30, text=text, token_count=len(text.split()),
                     is_sponsor=False, embedding=vector))
    await db.flush()


RETENTION = ("Onboarding is the most important lever for user retention, because "
             "every new user experiences it.")


@pytest.mark.slow
@requires_db
async def test_filter_applies_before_the_candidate_pool_is_cut(db):
    """More than a pool's worth of closer passages from other guests: a
    filter applied after the cut would find nothing for Casey."""
    crowd = await episode(db, "Other Guest")
    await chunks(db, crowd, [f"{RETENTION} Point {i}." for i in range(CANDIDATE_POOL + 5)])
    casey = await episode(db, "Casey Winters")
    await chunks(db, casey, ["Retention is the foundation of a growth loop; without it, "
                             "acquisition spend leaks out of the bucket."])
    await db.commit()

    unfiltered = await search(db, "how to improve user retention with onboarding", top_k=8)
    assert "Casey Winters" not in {c.guest for c in unfiltered.chunks}, "precondition"

    filtered = await search(db, "how to improve user retention with onboarding",
                            filters=SearchFilters(guest="casey"))
    assert {c.guest for c in filtered.chunks} == {"Casey Winters"}


@pytest.mark.slow
@requires_db
async def test_guest_match_is_partial_and_covers_joint_episodes(db):
    joint = await episode(db, "Aishwarya Reganti + Kiriti Badam")
    await chunks(db, joint, [RETENTION])
    await db.commit()
    found = await search(db, "user retention", filters=SearchFilters(guest="kiriti"))
    assert [c.guest for c in found.chunks] == ["Aishwarya Reganti + Kiriti Badam"]


@pytest.mark.slow
@requires_db
async def test_date_range(db):
    for year in (2022, 2023, 2024):
        await chunks(db, await episode(db, f"Guest {year}", date(year, 6, 1)), [RETENTION])
    await chunks(db, await episode(db, "Undated", None), [RETENTION])
    await db.commit()

    found = await search(db, "user retention", filters=SearchFilters.parse(since="2023", until="2023"))
    assert {c.guest for c in found.chunks} == {"Guest 2023"}

    found = await search(db, "user retention", filters=SearchFilters.parse(since="2023"))
    assert {c.guest for c in found.chunks} == {"Guest 2023", "Guest 2024"}, \
        "an episode with no publish date cannot satisfy a date range"


@pytest.mark.slow
@requires_db
async def test_unknown_guest_is_reported_as_such(db):
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await db.commit()
    result = await search(db, "user retention", filters=SearchFilters(guest="Elon"))
    assert (result.grounded, result.reason, result.chunks) == (False, "no_matching_episodes", [])


@pytest.mark.slow
@requires_db
async def test_a_wildcard_guest_matches_nobody(db):
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await db.commit()
    result = await search(db, "user retention", filters=SearchFilters(guest="%"))
    assert result.reason == "no_matching_episodes"


@pytest.mark.slow
@requires_db
async def test_filters_still_respect_the_grounding_floor(db):
    """The right guest talking about something else is still 'not covered'."""
    await chunks(db, await episode(db, "Casey Winters"), [RETENTION])
    await db.commit()
    result = await search(db, "best sourdough starter recipe",
                          filters=SearchFilters(guest="Casey"))
    assert result.grounded is False
    assert result.reason == "below_similarity_threshold"


# ------------------------------------------------------------------- API

@pytest.mark.slow
@pytest.mark.api
@requires_db
async def test_api_search_with_filters(sessionmaker):  # noqa: F811
    from app.db.session import get_db
    from app.main import create_app

    async with sessionmaker() as db:
        await chunks(db, await episode(db, "Casey Winters", date(2023, 1, 1)), [RETENTION])
        await chunks(db, await episode(db, "Adam Fishman", date(2021, 1, 1)), [RETENTION])
        await db.commit()

    async def override():
        async with sessionmaker() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override
    async with client_for(app) as client:
        response = await client.post("/api/search", json={
            "query": "user retention", "guest": "casey", "since": "2022-06-01"})
    assert response.status_code == 200
    assert {r["guest"] for r in response.json()["results"]} == {"Casey Winters"}


@pytest.mark.parametrize("body", [
    {"query": "q", "since": "2024-01-01", "until": "2023-01-01"},
    {"query": "q", "since": "2023"},          # the API takes full dates
    {"query": "q", "guest": "x" * 201},
])
@pytest.mark.api
async def test_api_search_filter_validation(body):
    from app.main import create_app

    async with client_for(create_app()) as client:
        response = await client.post("/api/search", json=body)
    assert response.status_code == 422
