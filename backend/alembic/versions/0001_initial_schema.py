"""Initial schema: knowledge base, sessions, artifacts, ingestion audit.

Revision ID: 0001
Revises:
Create Date: 2026-08-26
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    # pgvector must exist before any vector column is declared.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "episodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", sa.String(32), nullable=False, unique=True),
        sa.Column("guest", sa.String(300), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("youtube_url", sa.Text(), nullable=False),
        sa.Column("publish_date", sa.Date(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("keywords", postgresql.JSONB(), nullable=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_episodes_video_id", "episodes", ["video_id"])
    op.create_index("ix_episodes_content_hash", "episodes", ["content_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(200), nullable=True),
        sa.Column("start_seconds", sa.Integer(), nullable=True),
        sa.Column("end_seconds", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("is_sponsor", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index("ix_chunks_episode_id", "chunks", ["episode_id"])
    op.create_index("ix_chunks_is_sponsor", "chunks", ["is_sponsor"])
    op.create_index("ix_chunks_episode_ordinal", "chunks", ["episode_id", "ordinal"], unique=True)
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    # HNSW rather than IVFFlat: no training step, so it works on an empty table
    # and stays correct as rows are added incrementally during ingestion.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False, server_default="New chat"),
        sa.Column("user_metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(40), nullable=True),
        sa.Column("model", sa.String(120), nullable=True),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sanitizer_report", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_artifacts_session_id", "artifacts", ["session_id"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("episodes_seen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("episodes_ingested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("episodes_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunks_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingestion_runs")
    op.drop_table("artifacts")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("chunks")
    op.drop_table("episodes")
