"""Shared fixtures.

Database-backed tests run against a real Postgres with pgvector, because the
retrieval layer is written in Postgres-specific SQL (tsvector, `<=>`, FULL OUTER
JOIN) and a SQLite stand-in would test a different system than the one we ship.
When no database is reachable, those tests skip with a clear reason rather than
failing, so `pytest` is still useful on a laptop with nothing running.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import Base
from app.llm.base import ChatMessage, LLMProvider, LLMResponse, ProviderHealth, ToolCall, ToolSpec


@pytest.fixture(autouse=True)
def _trace_db_in_tmp(tmp_path, monkeypatch):
    """Every test that exercises the agent loop writes trace spans (see
    app/memory/trace.py). Without this, the default `/data/traces.db` path
    resolves on Windows to a real, absolute `C:\\data\\traces.db` and tests
    silently write real files onto the host outside the repo -- caught
    exactly that way once. Autouse so no test has to remember to opt in."""
    get_settings.cache_clear()
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "traces.db"))
    yield
    get_settings.cache_clear()

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://lenny:lenny_dev_password@localhost:5432/lenny_test",
)


def _database_available() -> bool:
    async def probe() -> bool:
        engine = create_async_engine(TEST_DATABASE_URL)
        try:
            async with engine.connect():
                return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    try:
        return asyncio.run(probe())
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _database_available(),
    reason=(
        "No test database. Start one with `docker compose up -d postgres` and create "
        "it with: docker compose exec postgres createdb -U lenny lenny_test"
    ),
)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    """A session against a freshly created schema, torn down afterwards."""
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


class FakeProvider(LLMProvider):
    """A scripted provider.

    Lets us assert on agent *routing* — did it search before answering, did the
    ungrounded guard fire — without a live model, which would make these tests
    slow and non-deterministic.
    """

    name = "fake"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.script:
            return LLMResponse(content="(exhausted)", provider=self.name, model=self.model)
        return self.script.pop(0)

    async def health(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider=self.name, model=self.model, detail="ok")


def text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, provider="fake", model="fake-model")


def tool_response(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:6]}", name=name, arguments=arguments)],
        provider="fake",
        model="fake-model",
    )
