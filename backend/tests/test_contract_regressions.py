"""Regressions for the defects contract testing found (agent-transcripts 14-17).

Schemathesis runs in CI against the live API; these pin each specific
finding in the fast suite, so a regression fails here first.
"""

from __future__ import annotations

import httpx
import pytest

from app.main import create_app
from tests.conftest import requires_db
from tests.test_api import client_for, sessionmaker  # noqa: F401 - fixture

pytestmark = pytest.mark.api


# ------------------------------------------------- 14: NUL characters -> 422

@pytest.mark.parametrize("method, path, body", [
    ("POST", "/api/sessions", {"title": "\x00"}),
    ("POST", "/api/sessions", {"title": "ok", "user_metadata": {"k": ["a\x00"]}}),
    ("POST", "/api/sessions", {"user_metadata": {"\x00": 1}}),
    ("POST", "/api/search", {"query": "retention\x00"}),
    ("POST", "/api/search", {"query": "retention", "guest": "\x00"}),
    ("POST", "/api/sessions/00000000-0000-0000-0000-000000000000/chat", {"message": "hi\x00"}),
], ids=["title", "metadata-value", "metadata-key", "query", "guest", "message"])
async def test_nul_characters_are_rejected_not_a_server_error(method, path, body):
    async with client_for(create_app()) as client:
        response = await client.request(method, path, json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "NUL" in error["fields"][0]["msg"]


# --------------------------------------- 15: request id on unhandled errors

async def test_an_unhandled_error_keeps_its_request_id():
    app = create_app()

    @app.get("/test-only/boom")
    async def boom():
        raise RuntimeError("unexpected")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test-only/boom", headers={"X-Request-ID": "trace-me-500"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["request_id"] == "trace-me-500"
    assert response.headers["X-Request-ID"] == "trace-me-500"


async def test_a_generated_request_id_matches_between_body_and_header():
    app = create_app()

    @app.get("/test-only/boom")
    async def boom():
        raise RuntimeError("unexpected")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test-only/boom")

    rid = response.json()["error"]["request_id"]
    assert rid != "-"
    assert response.headers["X-Request-ID"] == rid


# ------------------------------------------------ 16: offset beyond bigint

async def test_offset_past_postgres_bigint_is_422():
    async with client_for(create_app()) as client:
        response = await client.get("/api/sessions", params={"offset": 2**63})
    assert response.status_code == 422


@requires_db
async def test_offset_at_postgres_bigint_max_still_works(sessionmaker):  # noqa: F811
    from app.db.session import get_db

    async def override():
        async with sessionmaker() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override
    async with client_for(app) as client:
        response = await client.get("/api/sessions", params={"offset": 2**63 - 1})
    assert response.status_code == 200
    assert response.json() == []


# ------------------------------- 17: the schema documents the real errors

@pytest.fixture(scope="module")
def openapi() -> dict:
    return create_app().openapi()


def operations(openapi: dict):
    for path, ops in openapi["paths"].items():
        for method, op in ops.items():
            yield method.upper(), path, op


def test_fastapis_default_error_shape_is_gone(openapi):
    assert "HTTPValidationError" not in openapi["components"]["schemas"]


def test_every_422_documents_the_real_envelope(openapi):
    documented = [(m, p) for m, p, op in operations(openapi) if "422" in op["responses"]]
    assert documented, "precondition"
    for method, path, op in operations(openapi):
        if "422" in op["responses"]:
            ref = op["responses"]["422"]["content"]["application/json"]["schema"]["$ref"]
            assert ref.endswith("/ValidationErrorResponse"), (method, path)


@pytest.mark.parametrize("method, path", [
    ("GET", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("POST", "/api/sessions/{session_id}/chat"),
    ("POST", "/api/sessions/{session_id}/chat/stream"),
    ("GET", "/api/artifacts/{artifact_id}"),
])
def test_routes_that_can_404_say_so(openapi, method, path):
    assert "404" in openapi["paths"][path][method.lower()]["responses"]


def test_chat_documents_model_provider_errors(openapi):
    responses = openapi["paths"]["/api/sessions/{session_id}/chat"]["post"]["responses"]
    assert {"502", "503", "504"} <= set(responses)
