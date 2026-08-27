"""Request/response contracts.

Every endpoint validates in and out. Errors share one envelope so the frontend
has exactly one shape to handle, and every error carries a machine-readable
`code` plus a human `hint` that says what to actually do about it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- errors


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable description")
    hint: str = Field(default="", description="What the operator should do about it")
    request_id: str = Field(default="-")


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------- sessions


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    user_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class CitationOut(BaseModel):
    chunk_id: str
    episode_title: str
    guest: str
    timestamp: str = ""
    url: str = ""
    publish_date: str | None = None
    score: float = 0.0
    similarity: float = 0.0


class ArtifactSummary(BaseModel):
    id: uuid.UUID
    kind: Literal["markdown", "html"]
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ArtifactOut(ArtifactSummary):
    content: str
    sanitizer_report: dict[str, Any] | None = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    provider: str | None = None
    model: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    latency_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionDetail(SessionSummary):
    messages: list[MessageOut] = Field(default_factory=list)
    artifacts: list[ArtifactSummary] = Field(default_factory=list)


# ---------------------------------------------------------------- chat


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message cannot be blank")
        return stripped


class ToolCallOut(BaseModel):
    tool: str
    ok: bool
    latency_ms: int


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message: MessageOut
    artifacts: list[ArtifactOut] = Field(default_factory=list)
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    grounded: bool = Field(description="Whether the answer is backed by retrieved transcripts")
    provider: str
    model: str
    latency_ms: int


# ---------------------------------------------------------------- health


class ProviderStatus(BaseModel):
    provider: str
    model: str
    healthy: bool
    detail: str = ""
    latency_ms: int | None = None
    models_available: list[str] = Field(default_factory=list)


class KnowledgeBaseStatus(BaseModel):
    episodes: int
    chunks: int
    embedded_chunks: int
    ready: bool
    last_run_status: str | None = None
    last_run_finished_at: datetime | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str = "1.0.0"


class DeepHealthResponse(HealthResponse):
    database: dict[str, Any]
    providers: list[ProviderStatus]
    knowledge_base: KnowledgeBaseStatus
    config: dict[str, Any]


class ConfigResponse(BaseModel):
    """What the UI needs to render the provider badge and feature states."""

    provider: str
    model: str
    endpoint: str
    is_local: bool
    api_key_required: bool
    api_key_present: bool
    fallback_provider: str
    knowledge_base_ready: bool
    episodes: int
    chunks: int


# ---------------------------------------------------------------- search (debug)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=25)


class SearchResponse(BaseModel):
    query: str
    grounded: bool
    best_similarity: float
    latency_ms: int
    results: list[dict[str, Any]]
