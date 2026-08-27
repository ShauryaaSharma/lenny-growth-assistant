"""Agent routing and grounding behaviour.

These are the tests that matter most for the product's core promise. A RAG
assistant that *usually* searches before answering is not grounded — it is
grounded when it *cannot* answer without searching. That property is enforced
in `app.agent.runtime`, and this file is what holds it in place.

A scripted FakeProvider stands in for the model so the assertions are about
routing logic, not about whether a 3B model happened to behave this run.
"""

from __future__ import annotations

import uuid

import pytest

from app.agent import runtime as runtime_module
from app.agent.prompts import FORCE_SEARCH_NUDGE, is_trivial
from app.agent.runtime import run_agent
from app.rag import retriever as retriever_module
from app.rag.retriever import RetrievalResult, RetrievedChunk
from tests.conftest import FakeProvider, text_response, tool_response


def make_chunk(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=str(uuid.uuid4()),
        episode_id=str(uuid.uuid4()),
        text="Onboarding is the only part of your product everyone touches.",
        speaker="Adam Fishman",
        start_seconds=95,
        end_seconds=180,
        guest="Adam Fishman",
        title="How to build a high-performing growth team",
        youtube_url="https://www.youtube.com/watch?v=abc123",
        publish_date="2023-04-21",
        vector_similarity=0.78,
        text_rank=0.42,
        rrf_score=0.032,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


@pytest.fixture
def grounded_search(monkeypatch):
    """Retrieval that finds good material."""

    async def fake_search(db, query, top_k=None, min_similarity=None):
        return RetrievalResult(
            chunks=[make_chunk()], query=query, grounded=True, best_similarity=0.78
        )

    monkeypatch.setattr(retriever_module, "search", fake_search)
    import app.agent.tools as tools_module

    monkeypatch.setattr(tools_module, "search", fake_search)
    return fake_search


@pytest.fixture
def empty_search(monkeypatch):
    """Retrieval that finds nothing above the similarity floor."""

    async def fake_search(db, query, top_k=None, min_similarity=None):
        return RetrievalResult(
            chunks=[], query=query, grounded=False, reason="below_similarity_threshold"
        )

    import app.agent.tools as tools_module

    monkeypatch.setattr(tools_module, "search", fake_search)
    return fake_search


def install_provider(monkeypatch, provider: FakeProvider) -> None:
    async def fake_chat(messages, tools=None, temperature=0.3, max_tokens=None):
        return await provider.chat(messages, tools, temperature, max_tokens)

    monkeypatch.setattr(runtime_module, "chat_with_fallback", fake_chat)


class TestForcedRetrieval:
    async def test_answering_without_searching_is_rejected_once(
        self, monkeypatch, grounded_search
    ):
        """The core guarantee: the model does not get to skip retrieval."""
        provider = FakeProvider(
            [
                # First attempt: answers straight from parametric memory.
                text_response("Product-market fit is when customers pull the product from you."),
                # After the nudge it complies.
                tool_response("search_transcripts", query="signs of product-market fit"),
                text_response("Adam Fishman argues onboarding is the lever [1]."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "How do I know I have PMF?", [])

        assert "search_transcripts" in [t["tool"] for t in result.tool_calls]
        assert result.grounded is True
        assert result.content.startswith("Adam Fishman")
        nudges = [
            m for call in provider.calls for m in call if m.content == FORCE_SEARCH_NUDGE
        ]
        assert nudges, "the forced-retrieval nudge should have been injected"

    async def test_nudge_fires_at_most_once(self, monkeypatch, grounded_search):
        """A model that ignores the nudge must not trap the user in a loop."""
        provider = FakeProvider(
            [text_response("Still answering from memory."), text_response("Answering again.")]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "How do I price a B2B product?", [])
        assert result.content == "Answering again."
        assert result.iterations <= 3

    async def test_greetings_skip_retrieval(self, monkeypatch, grounded_search):
        """Forcing a corpus search on "hi" would be slow and absurd."""
        provider = FakeProvider([text_response("Hello. Ask me about product or growth.")])
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "hi", [])
        assert result.tool_calls == []
        assert result.content.startswith("Hello")


class TestUngroundedGuard:
    async def test_empty_retrieval_forces_an_admission(self, monkeypatch, empty_search):
        """With no evidence, the assistant must decline rather than improvise."""
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="how to bake sourdough"),
                # Model tries to answer anyway from general knowledge.
                text_response("Sourdough needs a starter and a long fermentation."),
                # After the guard it admits the gap.
                text_response("Lenny's Podcast transcripts don't cover sourdough baking."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "How do I bake sourdough?", [])
        assert "don't cover" in result.content
        assert result.citations == []
        # Regression: the guard must not report a refusal as a grounded answer
        # by reusing ctx.grounded as its own "already fired" flag.
        assert result.grounded is False

    async def test_search_tool_returns_an_explicit_refusal_instruction(
        self, monkeypatch, empty_search
    ):
        from app.agent.tools import ToolContext, execute_tool

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(ctx, "search_transcripts", {"query": "sourdough"})

        assert out["grounded"] is False
        assert out["results"] == []
        assert "do not answer" in out["instruction"].lower()
        assert ctx.grounded is False


class TestForcedArtifactCreation:
    """Guards against the failure observed live on llama3.2:3b: asked for "a
    one-page onboarding audit checklist," it searched correctly, then wrote the
    checklist as a plain chat message instead of calling create_artifact."""

    async def test_document_request_answered_in_prose_is_corrected(
        self, monkeypatch, grounded_search
    ):
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="onboarding audit"),
                # Model answers the checklist as chat prose -- no tool call.
                text_response("Here is a checklist:\n1. Map the signup flow\n2. ..."),
                # After the nudge it registers a real artifact instead.
                tool_response(
                    "create_artifact",
                    kind="markdown",
                    title="Onboarding Audit Checklist",
                    content="# Onboarding Audit Checklist\n\n- Map the signup flow",
                ),
                text_response("I've put together the checklist in the panel."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Build me a one-page onboarding audit checklist", []
        )

        assert len(result.artifacts) == 1
        assert result.artifacts[0].title == "Onboarding Audit Checklist"
        assert "panel" in result.content

    async def test_nudge_fires_at_most_once(self, monkeypatch, grounded_search):
        """A model that ignores the nudge twice must not trap the user in a loop."""
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="onboarding audit"),
                text_response("Here is a checklist in prose, still no artifact."),
                text_response("Still no artifact after the nudge."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Build me a one-page onboarding checklist", []
        )
        assert result.content == "Still no artifact after the nudge."
        assert result.artifacts == []

    async def test_ungrounded_document_request_is_not_forced_into_an_artifact(
        self, monkeypatch, empty_search
    ):
        """A request the corpus can't support should stay a refusal, not be
        coerced into producing an artifact anyway."""
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="sourdough checklist"),
                text_response("Here's a sourdough checklist from general knowledge."),
                text_response("Lenny's Podcast transcripts don't cover this topic."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Give me a one-page sourdough starter checklist", []
        )
        assert result.artifacts == []
        assert "don't cover" in result.content

    async def test_ordinary_questions_never_trigger_the_artifact_nudge(
        self, monkeypatch, grounded_search
    ):
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="pmf signals"),
                text_response("Adam Fishman argues PMF shows up as pull, not push [1]."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "What are signs of PMF?", [])
        assert result.artifacts == []
        assert result.content.startswith("Adam Fishman")


class TestToolDispatch:
    async def test_unknown_tool_is_reported_not_raised(self):
        from app.agent.tools import ToolContext, execute_tool

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(ctx, "delete_everything", {})
        assert "Unknown tool" in out["error"]
        assert ctx.tool_log[0]["ok"] is False

    async def test_artifact_tool_sanitises_before_registering(self):
        from app.agent.tools import ToolContext, execute_tool

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(
            ctx,
            "create_artifact",
            {"kind": "html", "title": "Report", "content": "<p>ok</p><script>alert(1)</script>"},
        )

        assert out["ok"] is True
        assert len(ctx.artifacts) == 1
        # The unsafe form must never reach storage.
        assert "<script" not in ctx.artifacts[0].content
        assert "dangerous_tag" in out["sanitizer_findings"]

    async def test_artifact_tool_rejects_empty_content(self):
        from app.agent.tools import ToolContext, execute_tool

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(ctx, "create_artifact", {"kind": "markdown", "title": "x", "content": "  "})
        assert "error" in out
        assert ctx.artifacts == []

    async def test_citations_are_deduplicated_across_searches(self, monkeypatch, grounded_search):
        """Two searches that hit the same passage must not cite it twice."""
        from app.agent.tools import ToolContext, execute_tool

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        shared = make_chunk(chunk_id="fixed-id")

        async def repeat_search(db, query, top_k=None, min_similarity=None):
            return RetrievalResult(chunks=[shared], query=query, grounded=True, best_similarity=0.8)

        import app.agent.tools as tools_module

        monkeypatch.setattr(tools_module, "search", repeat_search)

        await execute_tool(ctx, "search_transcripts", {"query": "a"})
        await execute_tool(ctx, "search_transcripts", {"query": "b"})
        assert len(ctx.citations) == 1


class TestTrivialDetection:
    @pytest.mark.parametrize("msg", ["hi", "Hello", "thanks!", "who are you", "ok"])
    def test_trivial_messages(self, msg: str):
        assert is_trivial(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "How do I improve retention?",
            "What did Adam Fishman say about onboarding?",
            "helpful frameworks for pricing",  # starts with "help" but is substantive
        ],
    )
    def test_substantive_messages(self, msg: str):
        assert is_trivial(msg) is False
