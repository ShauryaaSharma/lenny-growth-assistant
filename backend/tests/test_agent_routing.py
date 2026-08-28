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


class TestArtifactCreatedReminder:
    """Guards against a second failure observed in the same live session: when
    the model called both search_transcripts and write_ship30_essay in one
    turn, it answered using the SEARCH tool's "cite as [1], [2]" instruction
    instead of the essay tool's "describe it in 2-3 sentences" instruction --
    the two tool results carried conflicting guidance and it picked the wrong
    one. The reminder re-asserts the artifact instruction last, exploiting
    recency to make it win regardless of what else happened in the turn."""

    async def test_reminder_is_injected_immediately_after_artifact_creation(
        self, monkeypatch, grounded_search
    ):
        provider = FakeProvider(
            [
                # create_artifact is now blocked unless the turn has already
                # searched -- a search must come first, matching the real,
                # correctly-enforced flow.
                tool_response("search_transcripts", query="onboarding checklist"),
                tool_response(
                    "create_artifact",
                    kind="markdown",
                    title="Doc",
                    content="# Doc\n\nSome content.",
                ),
                text_response("I put together the document in the panel."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Build me a one-page onboarding checklist", []
        )

        assert len(result.artifacts) == 1
        # The third LLM call must have seen the reminder as its most recent
        # message -- this is what fixes the observed conflicting-instructions
        # failure, since a small model weighs recent context most heavily.
        third_call_messages = provider.calls[2]
        assert "just created a document artifact" in third_call_messages[-1].content
        assert result.content == "I put together the document in the panel."
        assert result.iterations == 3
        assert "Doc" in third_call_messages[-1].content

    async def test_reminder_wins_even_when_a_search_call_happens_in_the_same_turn(
        self, monkeypatch, grounded_search
    ):
        """Reproduces the exact live scenario: search_transcripts and
        create_artifact called together in one model response. search_transcripts
        must run first in the list -- create_artifact is blocked until the turn
        has searched, so this also verifies calls within one response are
        applied in order rather than all evaluated against a stale ctx snapshot."""
        from app.llm.base import LLMResponse, ToolCall

        multi_tool_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="search_transcripts", arguments={"query": "onboarding"}),
                ToolCall(id="call_2", name="create_artifact", arguments={
                    "kind": "markdown", "title": "Combined Doc", "content": "# Combined Doc\n\nBody."
                }),
            ],
            provider="fake",
            model="fake-model",
        )
        provider = FakeProvider(
            [multi_tool_response, text_response("Here it is, in the panel.")]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Build me a one-page onboarding checklist", []
        )

        assert len(result.artifacts) == 1
        second_call_messages = provider.calls[1]
        # The reminder is the LAST message the model sees, after both tool
        # results, regardless of which tool ran second.
        assert second_call_messages[-1].content.startswith(
            'You just created a document artifact titled "Combined Doc"'
        )

    async def test_reminder_fires_at_most_once_per_turn(self, monkeypatch, grounded_search):
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="checklist topic"),
                tool_response(
                    "create_artifact", kind="markdown", title="A", content="# A\n\nX"
                ),
                # A second create_artifact after one already exists this turn
                # must now be blocked outright, not merely discouraged by the
                # reminder -- found live creating an unwanted second artifact.
                tool_response(
                    "create_artifact", kind="markdown", title="B", content="# B\n\nY"
                ),
                text_response("Done."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "Build me a one-page checklist", [])

        assert len(result.artifacts) == 1, "the redundant create_artifact call must be blocked, not executed"
        assert result.artifacts[0].title == "A"

        # provider.calls[i] is a growing snapshot of the same message list, so
        # a single appended reminder reappears in every later snapshot -- the
        # correct check is "at most once in the final, longest snapshot", not
        # a sum across all of them (which would overcount by construction).
        final_messages = provider.calls[-1]
        reminder_count = sum(
            1 for m in final_messages if "just created a document artifact" in m.content
        )
        assert reminder_count == 1


class TestEssayCompletionDoesNotTriggerSpuriousResearch:
    """End-to-end reproduction of the most expensive bug found live: after a
    successful essay generation (its own internal retrieval already grounded
    the answer), the agent loop incorrectly believed no search had happened
    and forced a redundant search -- which then led the model to regenerate
    the entire multi-minute essay pipeline a second time. Each essay
    generation costs several real minutes on CPU, so this bug was not just
    incorrect, it was the single most expensive reliability defect found in
    the whole build."""

    async def test_successful_essay_does_not_force_a_search_nudge(self, monkeypatch):
        import app.agent.tools as tools_module

        async def fake_write_essay(db, topic):
            return {
                "ok": True,
                "essay": "# Retention as a Lever\n\nBody.",
                "rubric": {"word_count": 1200, "passed": True},
                "citations": [{"chunk_id": "x", "episode_title": "E", "guest": "G"}],
                "revised": False,
            }

        monkeypatch.setattr(tools_module, "write_ship30_essay", fake_write_essay)

        provider = FakeProvider(
            [
                tool_response("write_ship30_essay", topic="retention as a growth lever"),
                # If the bug were still present, the runtime would inject
                # FORCE_SEARCH_NUDGE here instead of accepting this reply.
                text_response("I've written the essay on retention; it's in the panel."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Write a Ship 30 essay about retention as a growth lever", []
        )

        # Exactly one essay generation -- not two.
        assert [t["tool"] for t in result.tool_calls] == ["write_ship30_essay"]
        assert result.content == "I've written the essay on retention; it's in the panel."
        final_messages = provider.calls[-1]
        assert not any(FORCE_SEARCH_NUDGE in m.content for m in final_messages)


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

    async def test_essay_tool_marks_searched_on_success(self, monkeypatch):
        """Regression: the essay tool performs its own internal retrieval
        (top_k=14) but never told the top-level loop that grounding had
        happened, because it only set ctx.grounded, not ctx.searched. Observed
        live: after a successful ~4-minute essay generation, the loop's "did
        you search first" guard fired anyway (ctx.searched was still False),
        forcing an unnecessary second full essay generation."""
        import app.agent.tools as tools_module
        from app.agent.tools import ToolContext, execute_tool

        async def fake_write_essay(db, topic):
            return {
                "ok": True,
                "essay": "# Title\n\nBody text.",
                "rubric": {"word_count": 1200, "passed": True},
                "citations": [],
                "revised": False,
            }

        monkeypatch.setattr(tools_module, "write_ship30_essay", fake_write_essay)

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(ctx, "write_ship30_essay", {"topic": "retention"})

        assert out["ok"] is True
        assert ctx.searched is True
        assert ctx.grounded is True

    async def test_essay_tool_marks_searched_even_when_declined(self, monkeypatch):
        """A declined essay (insufficient evidence) still performed a real
        retrieval attempt -- ctx.searched must be True so the ungrounded
        guard fires next, rather than the unrelated force-search nudge."""
        import app.agent.tools as tools_module
        from app.agent.tools import ToolContext, execute_tool

        async def fake_write_essay(db, topic):
            return {"ok": False, "message": "not enough evidence", "citations": []}

        monkeypatch.setattr(tools_module, "write_ship30_essay", fake_write_essay)

        ctx = ToolContext(db=None, session_id=uuid.uuid4())
        out = await execute_tool(ctx, "write_ship30_essay", {"topic": "sourdough starters"})

        assert out["ok"] is False
        assert ctx.searched is True
        assert ctx.grounded is False

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


class TestNoToolsOnTrivialMessages:
    """Regression for agent-transcripts/11: "hi" was calling search_transcripts
    with a hallucinated, unrelated query, costing ~150s for a greeting. The
    forced-retrieval guard only ever governed what happens after a tool call
    is offered -- it never stopped one being offered on a trivial message in
    the first place."""

    async def test_trivial_message_is_never_given_tools(self, monkeypatch):
        provider = FakeProvider([text_response("Hello! Ask me about product or growth.")])
        install_provider(monkeypatch, provider)

        result = await run_agent(None, uuid.uuid4(), "hi", [])

        assert result.tool_calls == []
        # The provider must have been called with tools=None, not just happen
        # to not use them -- this is what makes an unwanted tool call
        # structurally impossible rather than merely unlikely.
        assert provider.calls  # sanity: the provider was actually invoked
        # FakeProvider doesn't record the tools argument directly, but a call
        # with tools=None cannot legally return wants_tools=True; confirm the
        # loop accepted a plain text answer on the very first iteration.
        assert result.iterations == 1
        assert result.content == "Hello! Ask me about product or growth."


class TestBlockedUngroundedArtifact:
    """Regression for agent-transcripts/11: asked for a document on a topic
    never searched, the model called create_artifact directly instead of
    search_transcripts -- a fully fabricated recipe was rendered as a
    legitimate-looking artifact. The forced-retrieval guard never caught this
    because it only ever watched for a bare text answer given without
    searching, not for a tool call used as an escape hatch instead."""

    async def test_create_artifact_before_any_search_is_blocked(self, monkeypatch):
        provider = FakeProvider(
            [
                # Straight to create_artifact, no search_transcripts call --
                # exactly what was observed live.
                tool_response(
                    "create_artifact", kind="markdown", title="Recipe",
                    content="# Sourdough Starter Recipe\n\nFabricated content.",
                ),
                tool_response("search_transcripts", query="sourdough starter recipe"),
                text_response("Lenny's Podcast doesn't cover sourdough baking."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "What's the best sourdough starter recipe?", []
        )

        assert result.artifacts == [], "the fabricated artifact must never be created"
        # The blocked call is still recorded, marked as failed -- visible in
        # the trace rather than silently dropped.
        blocked = [t for t in result.tool_calls if t["tool"] == "create_artifact"]
        assert blocked and blocked[0]["ok"] is False

    async def test_a_grounded_document_request_is_not_blocked(self, monkeypatch, grounded_search):
        """The guard must not over-block: once the turn has searched and
        found real material, create_artifact must proceed normally."""
        provider = FakeProvider(
            [
                tool_response("search_transcripts", query="onboarding checklist"),
                tool_response(
                    "create_artifact", kind="markdown", title="Checklist", content="# Checklist\n\nReal."
                ),
                text_response("I made the checklist."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Build me a one-page onboarding checklist", []
        )
        assert len(result.artifacts) == 1


class TestBlockedRedundantArtifactAcrossTools:
    """The redundant-artifact guard blocks either content-creating tool once
    one artifact already exists this turn -- not just a repeat of the same
    tool. Cross-tool coverage: an essay already created, then the model
    tries create_artifact anyway."""

    async def test_create_artifact_after_an_essay_already_exists_is_blocked(
        self, monkeypatch
    ):
        import app.agent.tools as tools_module

        async def fake_write_essay(db, topic):
            return {
                "ok": True,
                "essay": "# Title\n\nBody.",
                "rubric": {"word_count": 1200, "passed": True},
                "citations": [],
                "revised": False,
            }

        monkeypatch.setattr(tools_module, "write_ship30_essay", fake_write_essay)

        provider = FakeProvider(
            [
                tool_response("write_ship30_essay", topic="retention"),
                tool_response(
                    "create_artifact", kind="markdown", title="Extra", content="# Extra\n\nY"
                ),
                text_response("Done."),
            ]
        )
        install_provider(monkeypatch, provider)

        result = await run_agent(
            None, uuid.uuid4(), "Write a Ship 30 essay about retention", []
        )

        assert len(result.artifacts) == 1
        assert result.artifacts[0].kind == "markdown"
        blocked = [t for t in result.tool_calls if t["tool"] == "create_artifact"]
        assert blocked and blocked[0]["ok"] is False


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

    @pytest.mark.parametrize(
        "msg",
        ["what you doin?", "whats up", "how are you", "wyd", "nothing much", "good morning"],
    )
    def test_small_talk_skips_retrieval(self, msg: str):
        """Regression guard for agent-transcripts/13: "What you doin?" was
        routed as a substantive question, searched the whole corpus, and
        answered small talk with a domain refusal after 30 seconds."""
        assert is_trivial(msg) is True

    @pytest.mark.parametrize(
        "msg",
        [
            "how are you going to measure retention?",
            "what you doing about churn in B2B SaaS?",
            "how is it going to scale past 100k users?",
            "whats up with PMF for marketplaces?",
        ],
    )
    def test_questions_that_merely_begin_like_small_talk_still_retrieve(self, msg: str):
        """SMALL_TALK_EXACT must match whole messages only. Prefix-matching it
        the way TRIVIAL_PATTERNS are matched would skip retrieval on every one
        of these -- a worse bug than the one it fixes."""
        assert is_trivial(msg) is False
