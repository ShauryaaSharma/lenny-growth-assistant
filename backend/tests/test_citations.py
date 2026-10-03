"""An answer carries the sources it actually cites -- not everything retrieved.

Found live: asked what Elon Musk said, the agent searched, found passages
from four unrelated guests above the relevance floor, correctly answered
that the transcripts had nothing from him, and the reply still listed all
four guests as its sources.
"""

from __future__ import annotations

import uuid

import pytest

import app.agent.tools as tools_module
from app.agent import runtime as runtime_module
from app.agent.runtime import run_agent
from app.agent.tools import ToolContext, cited_numbers, execute_tool
from app.rag.retriever import RetrievalResult
from tests.conftest import FakeProvider, text_response, tool_response
from tests.test_agent_routing import make_chunk

pytestmark = pytest.mark.unit

CHUNKS = {name: make_chunk(chunk_id=name, guest=f"Guest {name}") for name in "abcd"}


@pytest.mark.parametrize("text, numbers", [
    ("Retention compounds [1].", {1}),
    ("Both agree [1, 3].", {1, 3}),
    ("See [2-4] and [6].", {2, 3, 4, 6}),
    ("See [2–4].", {2, 3, 4}),          # en dash, as models write it
    ("Stacked [1][2].", {1, 2}),
    ("Between [2019-2023] growth slowed.", set()),   # a span of years, not sources
    ("No citations here.", set()),
    ("A link [here](https://x.com).", set()),
])
def test_cited_numbers(text, numbers):
    assert cited_numbers(text) == numbers


def test_numbers_across_texts_are_combined():
    assert cited_numbers("reply [1]", "document [3]") == {1, 3}


# ------------------------------------------------- numbering across a turn

@pytest.fixture
def searches(monkeypatch):
    """Each call returns the next list of chunks."""
    queue: list[list] = []

    async def fake_search(db, query, top_k=None, min_similarity=None, filters=None):
        return RetrievalResult(chunks=queue.pop(0), query=query, grounded=True,
                               best_similarity=0.8)

    monkeypatch.setattr(tools_module, "search", fake_search)
    return queue


async def test_a_passage_keeps_its_number_across_searches(searches):
    searches += [[CHUNKS["a"], CHUNKS["b"]], [CHUNKS["b"], CHUNKS["c"]]]
    ctx = ToolContext(db=None, session_id=uuid.uuid4())

    first = await execute_tool(ctx, "search_transcripts", {"query": "one"})
    second = await execute_tool(ctx, "search_transcripts", {"query": "two"})

    assert [(r["guest"], r["n"]) for r in first["results"]] == [("Guest a", 1), ("Guest b", 2)]
    assert [(r["guest"], r["n"]) for r in second["results"]] == [("Guest b", 2), ("Guest c", 3)]


async def test_cited_keeps_only_what_is_cited_with_its_number(searches):
    searches += [[CHUNKS["a"], CHUNKS["b"], CHUNKS["c"]]]
    ctx = ToolContext(db=None, session_id=uuid.uuid4())
    await execute_tool(ctx, "search_transcripts", {"query": "q"})

    cited = ctx.cited("Guest c argues this [3], building on [1]. Also [9].")

    assert [(c["chunk_id"], c["n"]) for c in cited] == [("a", 1), ("c", 3)]


async def test_a_reply_citing_nothing_carries_no_sources(searches):
    searches += [[CHUNKS["a"], CHUNKS["b"]]]
    ctx = ToolContext(db=None, session_id=uuid.uuid4())
    await execute_tool(ctx, "search_transcripts", {"query": "q"})
    assert ctx.cited("The transcripts don't cover this.") == []


async def test_markers_in_a_created_document_count(searches):
    searches += [[CHUNKS["a"], CHUNKS["b"]]]
    ctx = ToolContext(db=None, session_id=uuid.uuid4())
    await execute_tool(ctx, "search_transcripts", {"query": "q"})
    assert [c["n"] for c in ctx.cited("I made you a checklist.", "- Step one [2]")] == [2]


def test_an_essay_keeps_its_own_evidence_numbering():
    ctx = ToolContext(db=None, session_id=uuid.uuid4())
    ctx.essay_citations = [{"chunk_id": "x", "n": 1}, {"chunk_id": "y", "n": 2}]
    assert ctx.cited("Here is your essay.") == ctx.essay_citations


# ------------------------------------------------------ whole agent turns

def install(monkeypatch, *script) -> None:
    provider = FakeProvider(list(script))

    async def fake_chat(messages, tools=None, temperature=0.3, max_tokens=None):
        return await provider.chat(messages, tools, temperature, max_tokens)

    monkeypatch.setattr(runtime_module, "chat_with_fallback", fake_chat)


async def test_the_live_defect_a_refusal_after_a_grounded_search(monkeypatch, searches):
    searches += [[CHUNKS["a"], CHUNKS["b"], CHUNKS["c"], CHUNKS["d"]]]
    install(monkeypatch,
            tool_response("search_transcripts", query="Elon Musk growth"),
            text_response("The transcripts don't contain anything Elon Musk said about growth."))

    result = await run_agent(None, uuid.uuid4(), "What did Elon Musk say about growth?", [])

    assert result.grounded is True, "retrieval did clear the floor; that is not the bug"
    assert result.citations == []


async def test_a_cited_answer_keeps_exactly_its_sources(monkeypatch, searches):
    searches += [[CHUNKS["a"], CHUNKS["b"], CHUNKS["c"]]]
    install(monkeypatch,
            tool_response("search_transcripts", query="retention"),
            text_response("Guest b says retention compounds [2]."))

    result = await run_agent(None, uuid.uuid4(), "What about retention?", [])

    assert [(c["guest"], c["n"]) for c in result.citations] == [("Guest b", 2)]
