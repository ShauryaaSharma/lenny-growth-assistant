"""app/memory/: procedural memory, reducers, and the trace store.

Reducers are pure functions -- tested directly with no I/O. The trace store
touches SQLite, so those tests use a real temp file (fast, no external
service needed) rather than mocking sqlite3, since the whole point is
verifying the schema and queries actually work.
"""

from __future__ import annotations

import asyncio

import pytest

from app.llm.base import ChatMessage
from app.memory import trace
from app.memory.procedural import PROCEDURAL_MEMORY, render_primary_for_system_prompt
from app.memory.reducers import build_agent_messages, reduce_history, reduce_turn


class TestReduceHistory:
    def test_shorter_than_limit_is_unchanged(self):
        history = [ChatMessage(role="user", content="a")]
        assert reduce_history(history, max_messages=20) == history

    def test_longer_than_limit_keeps_only_the_most_recent(self):
        history = [ChatMessage(role="user", content=str(i)) for i in range(30)]
        windowed = reduce_history(history, max_messages=10)
        assert len(windowed) == 10
        assert windowed[0].content == "20"
        assert windowed[-1].content == "29"

    def test_zero_limit_keeps_nothing(self):
        history = [ChatMessage(role="user", content="a")]
        assert reduce_history(history, max_messages=0) == []

    def test_does_not_mutate_the_input(self):
        history = [ChatMessage(role="user", content=str(i)) for i in range(5)]
        original_len = len(history)
        reduce_history(history, max_messages=2)
        assert len(history) == original_len


class TestReduceTurn:
    def test_appends_user_message_only_when_no_reply_yet(self):
        merged = reduce_turn([], "hello", assistant_reply=None)
        assert len(merged) == 1
        assert merged[0].role == "user"
        assert merged[0].content == "hello"

    def test_appends_both_user_and_assistant_when_reply_present(self):
        merged = reduce_turn([], "hello", assistant_reply="hi there")
        assert [m.role for m in merged] == ["user", "assistant"]
        assert merged[1].content == "hi there"

    def test_accumulates_onto_existing_history(self):
        history = [ChatMessage(role="user", content="first")]
        merged = reduce_turn(history, "second", "second reply")
        assert len(merged) == 3
        assert history == [ChatMessage(role="user", content="first")], "input must not be mutated"


class TestBuildAgentMessages:
    def test_shape_is_system_then_history_then_new_user_message(self):
        history = [ChatMessage(role="user", content="q1"), ChatMessage(role="assistant", content="a1")]
        messages = build_agent_messages("SYSTEM", history, "q2", max_history_messages=20)
        assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[0].content == "SYSTEM"
        assert messages[-1].content == "q2"

    def test_windows_history_before_assembling(self):
        history = [ChatMessage(role="user", content=str(i)) for i in range(50)]
        messages = build_agent_messages("SYSTEM", history, "new", max_history_messages=4)
        # system + 4 windowed + new user message
        assert len(messages) == 6


class TestProceduralMemory:
    def test_at_least_one_principle_is_defined(self):
        assert len(PROCEDURAL_MEMORY) >= 1

    def test_every_principle_names_what_enforces_it(self):
        """A procedural-memory entry with no enforcement mechanism is just an
        aspiration, not a governing principle -- this keeps the file honest."""
        for p in PROCEDURAL_MEMORY:
            assert p.enforced_by.strip() != ""

    def test_primary_prompt_line_is_short(self):
        """Must stay a single terse sentence, not a paragraph -- this is the
        one principle actually injected into the live system prompt seen by
        a 3B model, which follows short imperative lines far more reliably
        than prose (the same reasoning behind every other rule in
        SYSTEM_PROMPT)."""
        line = render_primary_for_system_prompt()
        assert len(line) < 220
        assert "\n\n" not in line


class TestTraceStore:
    @pytest.fixture(autouse=True)
    def _isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(trace, "_db_path", lambda: tmp_path / "traces.db")

    async def test_record_and_read_a_span(self):
        span = trace.Span(
            session_id="s1", request_id="r1", kind="tool_call", name="search_transcripts",
            duration_ms=42, ok=True,
        )
        await trace.record_span(span)

        rows = await trace.read_spans(session_id="s1")
        assert len(rows) == 1
        assert rows[0]["name"] == "search_transcripts"
        assert rows[0]["duration_ms"] == 42
        assert rows[0]["ok"] == 1

    async def test_spans_are_scoped_by_session_id(self):
        await trace.record_span(trace.Span("s1", "r1", "tool_call", "a", 1))
        await trace.record_span(trace.Span("s2", "r2", "tool_call", "b", 1))

        assert len(await trace.read_spans(session_id="s1")) == 1
        assert len(await trace.read_spans(session_id="s2")) == 1

    async def test_spans_are_scoped_by_request_id_within_a_session(self):
        """request_id groups all spans of one turn -- two turns in the same
        session must not bleed into each other's trace."""
        await trace.record_span(trace.Span("s1", "turn-a", "llm_call", "ollama:x", 10))
        await trace.record_span(trace.Span("s1", "turn-a", "tool_call", "search", 5))
        await trace.record_span(trace.Span("s1", "turn-b", "llm_call", "ollama:x", 20))

        turn_a = await trace.read_spans(session_id="s1", request_id="turn-a")
        assert len(turn_a) == 2
        assert {r["name"] for r in turn_a} == {"ollama:x", "search"}

    async def test_a_failed_span_records_ok_false_and_meta(self):
        await trace.record_span(
            trace.Span("s1", "r1", "llm_call", "ollama:x", 0, ok=False, meta={"error": "timeout"})
        )
        rows = await trace.read_spans(session_id="s1")
        assert rows[0]["ok"] == 0
        assert "timeout" in rows[0]["meta"]

    async def test_a_write_failure_never_raises(self, monkeypatch):
        """Tracing must never be the reason a chat turn fails -- see the
        module docstring's rationale."""

        def _boom(_span):
            raise OSError("disk is full")

        monkeypatch.setattr(trace, "_write_span_sync", _boom)
        # Must not raise, despite the underlying write always failing.
        await trace.record_span(trace.Span("s1", "r1", "tool_call", "x", 1))

    def test_timer_measures_elapsed_milliseconds(self):
        with trace.Timer() as t:
            pass
        assert t.ms >= 0

    async def test_read_spans_on_empty_db_returns_empty_list(self):
        assert await trace.read_spans(session_id="nonexistent") == []
