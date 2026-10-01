"""HTTP-level API tests.

Every other test file calls functions directly. These go through the real
FastAPI app over httpx -- routing, request validation, dependency injection,
the error envelope, status codes and response models -- because that is the
contract the frontend actually depends on, and it can break without any
function-level test noticing.

Two tiers, marked accordingly:

* Contract tests need no database: validation, error envelopes, the liveness
  probe, and endpoints whose collaborators are swapped out.
* Integration tests run against real Postgres (see `requires_db`), with the
  app's `get_db` dependency pointed at the test schema. The agent is the only
  thing faked: a live model would make assertions about persistence depend on
  what a 3B model happened to say.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.runtime import AgentResult
from app.agent.tools import PendingArtifact
from app.api import routes_chat, routes_health
from app.db.models import Base, Message
from app.db.session import get_db
from app.llm.base import LLMUnavailableError, ProviderHealth
from app.main import create_app
from app.rag.retriever import RetrievalResult, RetrievedChunk
from app.schemas.api import KnowledgeBaseStatus
from tests.conftest import TEST_DATABASE_URL, requires_db

MISSING_ID = "00000000-0000-0000-0000-000000000000"


def assert_error(response: httpx.Response, status: int, code: str) -> dict:
    """Every error, from any route, must use the same typed envelope."""
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert set(error) >= {"code", "message", "hint", "request_id"}
    return error


def client_for(app) -> httpx.AsyncClient:
    # ASGITransport does not run the lifespan, so no model warm-up or corpus
    # seeding is triggered by constructing a client.
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture
async def api() -> AsyncGenerator[httpx.AsyncClient, None]:
    """A client for contract tests. No database behind it."""
    async with client_for(create_app()) as client:
        yield client


@pytest_asyncio.fixture
async def sessionmaker() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_api(sessionmaker) -> AsyncGenerator[httpx.AsyncClient, None]:
    """A client whose `get_db` is the test schema, with the same commit and
    rollback semantics as the production dependency."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    async with client_for(app) as client:
        yield client


def make_chunk(**overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=str(uuid.uuid4()),
        episode_id=str(uuid.uuid4()),
        text="Retention is the growth lever most teams ignore. " * 20,
        speaker="Guest",
        start_seconds=3725,
        end_seconds=3790,
        guest="Casey Winters",
        title="Growth loops",
        youtube_url="https://youtube.com/watch?v=abc",
        publish_date="2023-01-01",
        vector_similarity=0.712345,
        text_rank=0.0123456,
        rrf_score=0.0321987,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def scripted_agent(result: AgentResult | Exception):
    """Stand-in for `run_agent` that records what the route passed it."""
    calls: list[dict] = []

    async def fake_run_agent(db, session_id, user_message, history):
        calls.append({"session_id": session_id, "message": user_message, "history": history})
        if isinstance(result, Exception):
            raise result
        return result

    fake_run_agent.calls = calls
    return fake_run_agent


# ------------------------------------------------------------ contract tier


class TestHealthAndEnvelope:
    async def test_liveness_probe(self, api):
        response = await api.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_every_response_carries_a_request_id(self, api):
        response = await api.get("/health")
        assert len(response.headers["X-Request-ID"]) == 12

    async def test_caller_supplied_request_id_is_echoed(self, api):
        response = await api.get("/health", headers={"X-Request-ID": "trace-me-123"})
        assert response.headers["X-Request-ID"] == "trace-me-123"

    async def test_unknown_route_uses_the_error_envelope(self, api):
        assert_error(await api.get("/api/does-not-exist"), 404, "http_error")

    async def test_wrong_method_uses_the_error_envelope(self, api):
        assert_error(await api.put("/health"), 405, "http_error")

    async def test_openapi_schema_lists_every_route(self, api):
        paths = (await api.get("/openapi.json")).json()["paths"]
        assert set(paths) >= {
            "/health",
            "/health/deep",
            "/api/config",
            "/api/sessions",
            "/api/sessions/{session_id}",
            "/api/sessions/{session_id}/chat",
            "/api/sessions/{session_id}/trace",
            "/api/search",
            "/api/artifacts",
            "/api/artifacts/{artifact_id}",
        }


class TestRequestValidation:
    @pytest.mark.parametrize(
        "method, path, body",
        [
            ("get", "/api/sessions/not-a-uuid", None),
            ("delete", "/api/sessions/not-a-uuid", None),
            ("get", "/api/artifacts/not-a-uuid", None),
            ("get", "/api/sessions/not-a-uuid/trace", None),
            ("post", "/api/sessions/not-a-uuid/chat", {"message": "hi"}),
        ],
    )
    async def test_malformed_ids_are_rejected(self, api, method, path, body):
        kwargs = {"json": body} if body is not None else {}
        response = await api.request(method.upper(), path, **kwargs)
        error = assert_error(response, 422, "validation_error")
        assert any("session_id" in f["loc"] or "artifact_id" in f["loc"] for f in error["fields"])

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"message": ""},
            {"message": "   \n\t  "},
            {"message": "x" * 8001},
            {"message": 42},
        ],
        ids=["missing", "empty", "whitespace", "too-long", "wrong-type"],
    )
    async def test_chat_rejects_invalid_messages(self, api, body):
        response = await api.post(f"/api/sessions/{MISSING_ID}/chat", json=body)
        assert_error(response, 422, "validation_error")

    async def test_chat_accepts_the_maximum_length_boundary(self, api, monkeypatch):
        """8000 characters is valid, so the request must get past validation.
        The route body is stubbed to answer with a sentinel, which proves the
        request got past validation without needing a database."""

        async def reached_route(db, session_id):
            raise HTTPException(status_code=418, detail={"code": "reached_route", "message": ""})

        monkeypatch.setattr(routes_chat, "load_session", reached_route)
        response = await api.post(f"/api/sessions/{MISSING_ID}/chat", json={"message": "x" * 8000})
        assert_error(response, 418, "reached_route")

    @pytest.mark.parametrize(
        "body",
        [
            {"query": ""},
            {"query": "q" * 1001},
            {"query": "retention", "top_k": 0},
            {"query": "retention", "top_k": 26},
        ],
        ids=["empty-query", "query-too-long", "top_k-below-range", "top_k-above-range"],
    )
    async def test_search_rejects_invalid_requests(self, api, body):
        assert_error(await api.post("/api/search", json=body), 422, "validation_error")

    @pytest.mark.parametrize("query", ["limit=0", "limit=201", "offset=-1", "limit=abc"])
    async def test_session_list_rejects_bad_pagination(self, api, query):
        assert_error(await api.get(f"/api/sessions?{query}"), 422, "validation_error")

    async def test_session_title_is_length_limited(self, api):
        response = await api.post("/api/sessions", json={"title": "t" * 201})
        assert_error(response, 422, "validation_error")

    @pytest.mark.parametrize("query", ["limit=0", "limit=1001"])
    async def test_trace_rejects_bad_limits(self, api, query):
        response = await api.get(f"/api/sessions/{MISSING_ID}/trace?{query}")
        assert_error(response, 422, "validation_error")


class TestSearchEndpoint:
    async def test_shapes_and_rounds_retrieval_output(self, api, monkeypatch):
        chunk = make_chunk()
        captured = {}

        async def fake_search(db, query, top_k, filters=None):
            captured.update(query=query, top_k=top_k)
            return RetrievalResult(
                chunks=[chunk], query=query, grounded=True, best_similarity=0.712345, latency_ms=17
            )

        monkeypatch.setattr(routes_chat, "search", fake_search)
        response = await api.post("/api/search", json={"query": "retention", "top_k": 3})

        assert response.status_code == 200
        assert captured == {"query": "retention", "top_k": 3}
        body = response.json()
        assert body["grounded"] is True
        assert body["best_similarity"] == 0.7123
        assert body["latency_ms"] == 17

        [hit] = body["results"]
        assert hit["guest"] == "Casey Winters"
        assert hit["timestamp"] == "1:02:05"
        assert hit["url"] == "https://youtube.com/watch?v=abc&t=3725"
        assert hit["similarity"] == 0.7123
        assert hit["text_rank"] == 0.01235
        assert len(hit["excerpt"]) == 500

    async def test_defaults_top_k_to_eight(self, api, monkeypatch):
        captured = {}

        async def fake_search(db, query, top_k, filters=None):
            captured["top_k"] = top_k
            return RetrievalResult(query=query)

        monkeypatch.setattr(routes_chat, "search", fake_search)
        response = await api.post("/api/search", json={"query": "pmf"})
        assert response.status_code == 200
        assert captured["top_k"] == 8
        assert response.json()["results"] == []
        assert response.json()["grounded"] is False


class TestDeepHealth:
    """`/health/deep` reduces three independent probes to one status. Each
    probe is stubbed so every branch of that reduction is exercised."""

    @pytest.fixture
    def probes(self, monkeypatch):
        state = {"db": True, "provider_healthy": True, "kb_ready": True}

        async def check_database():
            return (True, "ok") if state["db"] else (False, "ConnectionRefusedError: down")

        async def health_all():
            settings = routes_health.get_settings()
            return [
                ProviderHealth(
                    healthy=state["provider_healthy"],
                    provider=settings.llm_provider,
                    model="test-model",
                    detail="ok" if state["provider_healthy"] else "connection refused",
                )
            ]

        async def kb_status(db):
            ready = state["kb_ready"]
            return KnowledgeBaseStatus(
                episodes=3 if ready else 0,
                chunks=90 if ready else 0,
                embedded_chunks=90 if ready else 0,
                ready=ready,
            )

        monkeypatch.setattr(routes_health, "check_database", check_database)
        monkeypatch.setattr(routes_health, "health_all", health_all)
        monkeypatch.setattr(routes_health, "_knowledge_base_status", kb_status)
        return state

    @pytest.mark.parametrize(
        "db, provider_healthy, kb_ready, expected",
        [
            (True, True, True, "ok"),
            (True, False, True, "degraded"),
            (True, True, False, "degraded"),
            (False, True, True, "error"),
            (False, False, False, "error"),
        ],
    )
    async def test_status_reduction(self, api, probes, db, provider_healthy, kb_ready, expected):
        probes.update(db=db, provider_healthy=provider_healthy, kb_ready=kb_ready)
        response = await api.get("/health/deep")

        assert response.status_code == 200, "the probe must answer even when unhealthy"
        body = response.json()
        assert body["status"] == expected
        assert body["database"]["healthy"] is db
        assert body["providers"][0]["healthy"] is provider_healthy
        assert body["knowledge_base"]["ready"] is kb_ready

    async def test_broken_knowledge_base_query_does_not_break_the_probe(
        self, api, probes, monkeypatch
    ):
        async def exploding_kb(db):
            raise RuntimeError("relation chunks does not exist")

        monkeypatch.setattr(routes_health, "_knowledge_base_status", exploding_kb)
        response = await api.get("/health/deep")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["knowledge_base"]["ready"] is False
        assert "relation chunks does not exist" in body["knowledge_base"]["last_run_status"]

    async def test_reports_retrieval_config(self, api, probes):
        config = (await api.get("/health/deep")).json()["config"]
        assert {"embedding_model", "retrieval_top_k", "retrieval_min_similarity"} <= set(config)


class TestTraceEndpoint:
    async def test_session_without_history_returns_empty_list(self, api):
        response = await api.get(f"/api/sessions/{uuid.uuid4()}/trace")
        assert response.status_code == 200
        assert response.json() == []


# --------------------------------------------------------- integration tier


@requires_db
class TestSessionLifecycle:
    async def test_create_returns_201_with_defaults(self, db_api):
        response = await db_api.post("/api/sessions", json={})
        assert response.status_code == 201
        body = response.json()
        uuid.UUID(body["id"])
        assert body["title"] == "New chat"
        assert body["message_count"] == 0

    async def test_create_honours_a_title(self, db_api):
        response = await db_api.post("/api/sessions", json={"title": "Pricing research"})
        assert response.json()["title"] == "Pricing research"

    async def test_created_session_is_readable(self, db_api):
        created = (await db_api.post("/api/sessions", json={"title": "Read me"})).json()
        response = await db_api.get(f"/api/sessions/{created['id']}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == created["id"]
        assert body["messages"] == []
        assert body["artifacts"] == []

    async def test_missing_session_is_a_typed_404(self, db_api):
        error = assert_error(
            await db_api.get(f"/api/sessions/{MISSING_ID}"), 404, "session_not_found"
        )
        assert MISSING_ID in error["message"]

    async def test_delete_returns_204_then_the_session_is_gone(self, db_api):
        created = (await db_api.post("/api/sessions", json={})).json()

        response = await db_api.delete(f"/api/sessions/{created['id']}")
        assert response.status_code == 204
        assert response.content == b""

        assert_error(await db_api.get(f"/api/sessions/{created['id']}"), 404, "session_not_found")

    async def test_deleting_a_missing_session_is_a_404(self, db_api):
        assert_error(
            await db_api.delete(f"/api/sessions/{MISSING_ID}"), 404, "session_not_found"
        )

    async def test_list_is_newest_first_and_paginates(self, db_api):
        ids = [
            (await db_api.post("/api/sessions", json={"title": f"s{i}"})).json()["id"]
            for i in range(3)
        ]

        everything = (await db_api.get("/api/sessions")).json()
        assert [s["id"] for s in everything] == list(reversed(ids))

        page = (await db_api.get("/api/sessions?limit=1&offset=1")).json()
        assert [s["id"] for s in page] == [ids[1]]


@requires_db
class TestChatOverHttp:
    async def test_full_turn_is_persisted_and_returned(self, db_api, sessionmaker, monkeypatch):
        citation = make_chunk().as_citation()
        agent = scripted_agent(
            AgentResult(
                content="Retention compounds; acquisition does not.",
                citations=[citation],
                artifacts=[
                    PendingArtifact(
                        id="pending-1",
                        kind="markdown",
                        title="Retention checklist",
                        content="# Checklist",
                        sanitizer_report={"removed": []},
                    )
                ],
                tool_calls=[{"tool": "search_transcripts", "ok": True, "latency_ms": 40}],
                provider="fake",
                model="fake-model",
                latency_ms=1234,
                grounded=True,
            )
        )
        monkeypatch.setattr(routes_chat, "run_agent", agent)
        session_id = (await db_api.post("/api/sessions", json={})).json()["id"]

        response = await db_api.post(
            f"/api/sessions/{session_id}/chat", json={"message": "  Why does retention matter?  "}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == session_id
        assert body["grounded"] is True
        assert body["message"]["role"] == "assistant"
        assert body["message"]["citations"][0]["guest"] == "Casey Winters"
        assert body["tool_calls"] == [{"tool": "search_transcripts", "ok": True, "latency_ms": 40}]
        [artifact] = body["artifacts"]
        assert artifact["title"] == "Retention checklist"

        # The request schema strips whitespace before the agent sees it.
        assert agent.calls[0]["message"] == "Why does retention matter?"
        assert agent.calls[0]["history"] == []

        # Both turns are persisted, in order, and the first message named the chat.
        detail = (await db_api.get(f"/api/sessions/{session_id}")).json()
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
        assert detail["title"] == "Why does retention matter?"
        assert detail["message_count"] == 2
        assert [a["id"] for a in detail["artifacts"]] == [artifact["id"]]

        # The artifact is served in full, and listed under its own session only.
        full = (await db_api.get(f"/api/artifacts/{artifact['id']}")).json()
        assert full["content"] == "# Checklist"
        listed = (await db_api.get(f"/api/artifacts?session_id={session_id}")).json()
        assert [a["id"] for a in listed] == [artifact["id"]]
        other = (await db_api.post("/api/sessions", json={})).json()["id"]
        assert (await db_api.get(f"/api/artifacts?session_id={other}")).json() == []

    async def test_second_turn_receives_prior_history(self, db_api, monkeypatch):
        agent = scripted_agent(AgentResult(content="answer", provider="fake", model="fake-model"))
        monkeypatch.setattr(routes_chat, "run_agent", agent)
        session_id = (await db_api.post("/api/sessions", json={})).json()["id"]

        await db_api.post(f"/api/sessions/{session_id}/chat", json={"message": "first"})
        await db_api.post(f"/api/sessions/{session_id}/chat", json={"message": "second"})

        history = agent.calls[1]["history"]
        assert [(m.role, m.content) for m in history] == [("user", "first"), ("assistant", "answer")]

    async def test_history_is_isolated_between_sessions(self, db_api, monkeypatch):
        agent = scripted_agent(AgentResult(content="answer", provider="fake", model="fake-model"))
        monkeypatch.setattr(routes_chat, "run_agent", agent)
        a = (await db_api.post("/api/sessions", json={})).json()["id"]
        b = (await db_api.post("/api/sessions", json={})).json()["id"]

        await db_api.post(f"/api/sessions/{a}/chat", json={"message": "only in A"})
        await db_api.post(f"/api/sessions/{b}/chat", json={"message": "only in B"})

        assert agent.calls[1]["history"] == []

    async def test_long_first_message_is_truncated_into_the_title(self, db_api, monkeypatch):
        monkeypatch.setattr(
            routes_chat, "run_agent", scripted_agent(AgentResult(content="ok", provider="f", model="m"))
        )
        session_id = (await db_api.post("/api/sessions", json={})).json()["id"]

        await db_api.post(f"/api/sessions/{session_id}/chat", json={"message": "word " * 40})

        title = (await db_api.get(f"/api/sessions/{session_id}")).json()["title"]
        assert len(title) == routes_chat.TITLE_MAX + 3
        assert title.endswith("...")

    async def test_chat_on_a_missing_session_is_a_404(self, db_api, monkeypatch):
        agent = scripted_agent(AgentResult(content="never", provider="f", model="m"))
        monkeypatch.setattr(routes_chat, "run_agent", agent)

        response = await db_api.post(f"/api/sessions/{MISSING_ID}/chat", json={"message": "hi"})

        assert_error(response, 404, "session_not_found")
        assert agent.calls == [], "the agent must not run for a session that does not exist"

    async def test_llm_outage_is_typed_and_keeps_the_users_turn(
        self, db_api, sessionmaker, monkeypatch
    ):
        """The route commits the user's message before running the agent, so a
        model outage leaves a resumable conversation instead of losing the turn."""
        monkeypatch.setattr(
            routes_chat, "run_agent", scripted_agent(LLMUnavailableError("ollama is not running"))
        )
        session_id = (await db_api.post("/api/sessions", json={})).json()["id"]

        response = await db_api.post(f"/api/sessions/{session_id}/chat", json={"message": "hello?"})

        error = assert_error(response, 503, "llm_unavailable")
        assert "ollama" in error["hint"]
        async with sessionmaker() as db:
            stored = (
                await db.execute(select(Message).where(Message.session_id == uuid.UUID(session_id)))
            ).scalars().all()
        assert [(m.role, m.content) for m in stored] == [("user", "hello?")]


@requires_db
class TestArtifactsAndConfig:
    async def test_missing_artifact_is_a_typed_404(self, db_api):
        assert_error(
            await db_api.get(f"/api/artifacts/{MISSING_ID}"), 404, "artifact_not_found"
        )

    async def test_empty_listing(self, db_api):
        response = await db_api.get("/api/artifacts")
        assert response.status_code == 200
        assert response.json() == []

    async def test_config_reports_an_empty_knowledge_base(self, db_api):
        response = await db_api.get("/api/config")
        assert response.status_code == 200
        body = response.json()
        assert body["knowledge_base_ready"] is False
        assert body["episodes"] == 0
        assert body["chunks"] == 0
        assert body["provider"]
