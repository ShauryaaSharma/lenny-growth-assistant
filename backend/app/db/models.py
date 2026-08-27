"""SQLAlchemy models — the full persistence schema.

Design notes:
  * `episodes` / `chunks` hold the knowledge base. A chunk always points back to
    its episode and carries a timestamp, so every citation can deep-link to the
    exact moment on YouTube.
  * `chunks.tsv` is a generated tsvector column, letting Postgres do lexical
    search natively alongside pgvector's semantic search (hybrid retrieval).
  * `sessions` / `messages` hold conversations. Sessions are independent by
    construction: retrieval and history are always scoped by session_id.
  * `ingestion_runs` makes the "how is the KB refreshed?" question auditable.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384  # keep in sync with settings.embedding_dim (see alembic migration)


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    video_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    guest: Mapped[str] = mapped_column(String(300))
    title: Mapped[str] = mapped_column(Text)
    youtube_url: Mapped[str] = mapped_column(Text)
    publish_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    source_path: Mapped[str] = mapped_column(Text)
    # Content hash makes re-ingestion idempotent: unchanged episodes are skipped.
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="episode", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str | None] = mapped_column(String(200), nullable=True)
    start_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    # Sponsor reads are ingested but excluded from retrieval — see rag/chunking.py.
    is_sponsor: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    episode: Mapped[Episode] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_episode_ordinal", "episode_id", "ordinal", unique=True),
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    # Free-form client metadata (display name, locale, feature flags...).
    user_metadata: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text)
    # Denormalised so the transcript of a conversation stays truthful about which
    # model produced which turn, even after the operator flips the provider.
    provider: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    citations: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[Session] = relationship(back_populates="messages")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(20))  # markdown | html
    title: Mapped[str] = mapped_column(String(300))
    # Always the post-sanitisation content. The raw model output is never stored,
    # so a later bug in the viewer cannot resurrect an unsafe payload.
    content: Mapped[str] = mapped_column(Text)
    sanitizer_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[str] = mapped_column(String(20), default="running")  # running|ok|failed
    source: Mapped[str] = mapped_column(Text)
    episodes_seen: Mapped[int] = mapped_column(Integer, default=0)
    episodes_ingested: Mapped[int] = mapped_column(Integer, default=0)
    episodes_skipped: Mapped[int] = mapped_column(Integer, default=0)
    chunks_written: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
