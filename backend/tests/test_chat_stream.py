"""Streaming chat: progress events while the agent works, the answer only
once the guards have accepted it, and the same persistence as `/chat`.

The agent loop here is the real one (app.agent.runtime) driven by the
scripted FakeProvider, so the events asserted are the ones a user would see,
not ones a stub chose to send.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.agent import runtime as runtime_module
from app.agent.prompts import FORCE_SEARCH_NUDGE
from app.agent.runtime import run_agent
from app.api import routes_chat
from app.db.models import Message
from app.db.session import get_db, get_session_factory
from app.llm.base import LLMUnavailableError
from app.main import create_app
from tests.conftest import FakeProvider, requires_db, text_response, tool_response
from tests.test_agent_routing import empty_search, grounded_search  # noqa: F401 - fixtures
from tests.test_api import client_for, sessionmaker  # noqa: F401 - fixture

MISSING_ID = "00000000-0000-0000-0000-000000000000"


def install(monkeypatch, *script) -> FakeProvider:
    provider = FakeProvider(list(script))

    async def fake_chat(messages, tools=None, temperature=0.3, max_tokens=None):
        return await provider.chat(messages, tools, temperature, max_tokens)

    monkeypatch.setattr(runtime_module, "chat_with_fallback", fake_chat)
    return provider


def parse_sse(text: str) -> list[dict]:
    """Events in order; keepalive comments are kept as {"type": "keepalive"}."""
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            events.append({"type": "keepalive"})
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        event = json.loads(lines["data"])
        assert event["type"] == lines["event"]
        events.append(event)
    return events


# ------------------------------------------------- runtime: what is emitted

async def run_collecting(message: str) -> tuple[list[dict], object]:
    events: list[dict] = []

    async def sink(event):
        events.append(event)

    result = await run_agent(None, uuid.uuid4(), message, [], on_event=sink)
    return events, result


async def test_a_grounded_turn_reports_each_step(monkeypatch, grounded_search):  # noqa: F811
    install(monkeypatch,
            tool_response("search_transcripts", query="retention benchmarks"),
            text_response("Retention compounds [1]."))

    events, result = await run_collecting("What is a good retention benchmark?")

    assert [e["type"] for e in events] == ["thinking", "tool_start", "tool_end", "thinking"]
    assert events[1] == {"type": "tool_start", "tool": "search_transcripts",
                         "detail": "retention benchmarks"}
    assert events[2]["summary"] == "1 passage from 1 episode"
    assert events[2]["ok"] is True
    assert result.content == "Retention compounds [1]."


async def test_guards_are_reported(monkeypatch, grounded_search):  # noqa: F811
    provider = install(monkeypatch,
                       text_response("PMF is when customers pull."),       # skips search
                       tool_response("search_transcripts", query="pmf"),
                       text_response("Adam Fishman says [1]."))

    events, _ = await run_collecting("How do I know I have PMF?")

    assert {"type": "guard", "guard": "forced_retrieval"} in events
    assert any(m.content == FORCE_SEARCH_NUDGE for call in provider.calls for m in call)


async def test_an_empty_search_is_summarised_honestly(monkeypatch, empty_search):  # noqa: F811
    install(monkeypatch,
            tool_response("search_transcripts", query="sourdough"),
            text_response("I made a recipe."),
            text_response("The transcripts don't cover sourdough."))

    events, _ = await run_collecting("What's the best sourdough starter?")

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["summary"] == "nothing relevant found"
    assert {"type": "guard", "guard": "ungrounded"} in events


async def test_draft_answers_never_appear_in_events(monkeypatch, grounded_search):  # noqa: F811
    """The rejected draft is exactly what the guards exist to hide."""
    install(monkeypatch,
            text_response("FABRICATED DRAFT"),
            tool_response("search_transcripts", query="pmf"),
            text_response("Grounded answer [1]."))

    events, _ = await run_collecting("How do I know I have PMF?")

    assert "FABRICATED DRAFT" not in json.dumps(events)
    assert "Grounded answer" not in json.dumps(events)


async def test_a_failing_sink_does_not_break_the_turn(monkeypatch, grounded_search):  # noqa: F811
    install(monkeypatch,
            tool_response("search_transcripts", query="q"),
            text_response("Answer [1]."))

    async def broken_sink(event):
        raise ConnectionResetError("client went away")

    result = await run_agent(None, uuid.uuid4(), "What about retention?", [], on_event=broken_sink)
    assert result.content == "Answer [1]."


async def test_no_sink_is_the_old_behaviour(monkeypatch, grounded_search):  # noqa: F811
    install(monkeypatch,
            tool_response("search_transcripts", query="q"),
            text_response("Answer [1]."))
    result = await run_agent(None, uuid.uuid4(), "What about retention?", [])
    assert result.content == "Answer [1]."


# ---------------------------------------------------- endpoint, over HTTP

@pytest_asyncio.fixture
async def stream_api(sessionmaker):  # noqa: F811
    async def override_get_db():
        async with sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_session_factory] = lambda: sessionmaker
    async with client_for(app) as client:
        yield client


async def new_session(client) -> str:
    return (await client.post("/api/sessions", json={})).json()["id"]


async def stream(client, session_id: str, message: str = "What about retention?"):
    response = await client.post(f"/api/sessions/{session_id}/chat/stream",
                                 json={"message": message})
    return response, parse_sse(response.text) if response.status_code == 200 else []


@requires_db
async def test_endpoint_streams_progress_then_the_saved_turn(
        stream_api, monkeypatch, grounded_search):  # noqa: F811
    install(monkeypatch,
            tool_response("search_transcripts", query="retention"),
            text_response("Retention compounds [1]."))
    session_id = await new_session(stream_api)

    response, events = await stream(stream_api, session_id)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in events] == ["thinking", "tool_start", "tool_end", "thinking", "done"]
    done = events[-1]["response"]
    assert done["message"]["content"] == "Retention compounds [1]."
    assert done["grounded"] is True

    detail = (await stream_api.get(f"/api/sessions/{session_id}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["id"] == done["message"]["id"]
    assert detail["title"] == "What about retention?"


@requires_db
async def test_done_matches_what_the_non_streaming_endpoint_returns(
        stream_api, monkeypatch, grounded_search):  # noqa: F811
    script = [tool_response("search_transcripts", query="retention"),
              text_response("Retention compounds [1].")]
    install(monkeypatch, *script)
    _, events = await stream(stream_api, await new_session(stream_api))

    install(monkeypatch, *script)
    plain = (await stream_api.post(f"/api/sessions/{await new_session(stream_api)}/chat",
                                   json={"message": "What about retention?"})).json()

    streamed = events[-1]["response"]
    assert set(streamed) == set(plain)
    for key in ("grounded", "provider", "model"):
        assert streamed[key] == plain[key], key
    assert [t["tool"] for t in streamed["tool_calls"]] == [t["tool"] for t in plain["tool_calls"]]
    assert streamed["message"]["content"] == plain["message"]["content"]
    # The fake retriever mints a fresh chunk id per call; everything else must agree.
    def without_ids(citations):
        return [{k: v for k, v in c.items() if k != "chunk_id"} for c in citations]

    assert without_ids(streamed["message"]["citations"]) == without_ids(plain["message"]["citations"])


@requires_db
async def test_llm_outage_ends_the_stream_with_a_typed_error_and_keeps_the_turn(
        stream_api, sessionmaker, monkeypatch):  # noqa: F811
    async def down(*a, **k):
        raise LLMUnavailableError("ollama is not running")

    monkeypatch.setattr(runtime_module, "chat_with_fallback", down)
    session_id = await new_session(stream_api)

    _, events = await stream(stream_api, session_id)

    error = events[-1]
    assert (error["type"], error["status"], error["error"]["code"]) == (
        "error", 503, "llm_unavailable")
    assert "ollama" in error["error"]["hint"]
    async with sessionmaker() as db:
        stored = (await db.execute(
            select(Message).where(Message.session_id == uuid.UUID(session_id)))).scalars().all()
    assert [(m.role, m.content) for m in stored] == [("user", "What about retention?")]


@requires_db
async def test_an_unexpected_failure_still_ends_the_stream(stream_api, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(routes_chat, "run_agent", boom)
    _, events = await stream(stream_api, await new_session(stream_api))
    assert events[-1]["error"]["code"] == "internal_error"


@requires_db
async def test_keepalives_are_sent_while_the_model_is_slow(stream_api, monkeypatch):
    monkeypatch.setattr(routes_chat, "KEEPALIVE_SECONDS", 0.05)

    async def slow_agent(db, session_id, message, history, on_event=None):
        await asyncio.sleep(0.2)
        return runtime_module.AgentResult(content="late", provider="fake", model="m")

    monkeypatch.setattr(routes_chat, "run_agent", slow_agent)
    _, events = await stream(stream_api, await new_session(stream_api))

    assert {"type": "keepalive"} in events
    assert events[-1]["type"] == "done"


@requires_db
async def test_missing_session_is_a_plain_404_not_a_stream(stream_api):
    response = await stream_api.post(f"/api/sessions/{MISSING_ID}/chat/stream",
                                     json={"message": "hi"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"


@pytest.mark.parametrize("body", [{}, {"message": "   "}, {"message": "x" * 8001}])
async def test_invalid_messages_are_rejected_before_streaming(body):
    async with client_for(create_app()) as client:
        response = await client.post(f"/api/sessions/{MISSING_ID}/chat/stream", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@requires_db
async def test_the_turn_is_saved_even_if_nobody_reads_the_stream(
        stream_api, sessionmaker, monkeypatch, grounded_search):  # noqa: F811
    """A closed tab must not lose the answer: the turn runs as its own task."""
    install(monkeypatch,
            tool_response("search_transcripts", query="retention"),
            text_response("Saved anyway [1]."))
    session_id = await new_session(stream_api)

    async with stream_api.stream("POST", f"/api/sessions/{session_id}/chat/stream",
                                 json={"message": "What about retention?"}) as response:
        assert response.status_code == 200
        # Leave without reading a single event.

    await asyncio.gather(*list(routes_chat._running_turns))
    async with sessionmaker() as db:
        stored = (await db.execute(
            select(Message).where(Message.session_id == uuid.UUID(session_id))
            .order_by(Message.created_at))).scalars().all()
    assert [m.content for m in stored] == ["What about retention?", "Saved anyway [1]."]
