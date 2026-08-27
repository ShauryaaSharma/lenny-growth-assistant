"""Session isolation and persistence.

The product claim is "each session must maintain independent context." That is
only true if it holds at the database query level, not just by convention in
application code that happens to always pass the right id. These tests attack
that boundary directly.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.db.models import Artifact, Message, Session
from tests.conftest import requires_db

pytestmark = requires_db


class TestSessionIsolation:
    async def test_messages_are_scoped_to_their_session(self, db):
        a = Session(title="Session A")
        b = Session(title="Session B")
        db.add_all([a, b])
        await db.flush()

        db.add(Message(session_id=a.id, role="user", content="What is PMF?"))
        db.add(Message(session_id=b.id, role="user", content="How do I hire a PM?"))
        await db.commit()

        a_messages = (
            await db.execute(select(Message).where(Message.session_id == a.id))
        ).scalars().all()
        b_messages = (
            await db.execute(select(Message).where(Message.session_id == b.id))
        ).scalars().all()

        assert [m.content for m in a_messages] == ["What is PMF?"]
        assert [m.content for m in b_messages] == ["How do I hire a PM?"]

    async def test_deleting_a_session_cascades_its_messages(self, db):
        s = Session(title="Ephemeral")
        db.add(s)
        await db.flush()
        db.add(Message(session_id=s.id, role="user", content="hello"))
        await db.commit()

        await db.delete(s)
        await db.commit()

        remaining = (
            await db.execute(select(Message).where(Message.session_id == s.id))
        ).scalars().all()
        assert remaining == [], "orphaned messages after session delete"

    async def test_deleting_a_session_cascades_its_artifacts(self, db):
        s = Session(title="Has an artifact")
        db.add(s)
        await db.flush()
        db.add(
            Artifact(session_id=s.id, kind="markdown", title="Doc", content="# Doc")
        )
        await db.commit()

        await db.delete(s)
        await db.commit()

        remaining = (
            await db.execute(select(Artifact).where(Artifact.session_id == s.id))
        ).scalars().all()
        assert remaining == []

    async def test_deleting_a_message_preserves_its_artifact(self, db):
        """Artifact.message_id is ON DELETE SET NULL, not CASCADE -- a document
        the user is looking at should survive if the triggering message is
        ever pruned, rather than vanishing out from under them."""
        s = Session(title="s")
        db.add(s)
        await db.flush()
        m = Message(session_id=s.id, role="assistant", content="here is your doc")
        db.add(m)
        await db.flush()
        art = Artifact(session_id=s.id, message_id=m.id, kind="markdown", title="D", content="# D")
        db.add(art)
        await db.commit()

        await db.delete(m)
        await db.commit()

        # The DB enforces ON DELETE SET NULL itself, but the session's identity
        # map still holds the pre-delete Python object unless told otherwise --
        # db.get() would silently return stale data without an explicit,
        # awaited refresh (a bare expire() defers the reload to attribute
        # access, which async SQLAlchemy cannot do implicitly).
        await db.refresh(art)
        assert art.message_id is None

    async def test_two_sessions_can_share_no_state_even_with_colliding_content(self, db):
        """Identical message text in two sessions must not merge or conflate."""
        a, b = Session(title="A"), Session(title="B")
        db.add_all([a, b])
        await db.flush()
        db.add(Message(session_id=a.id, role="user", content="same question"))
        db.add(Message(session_id=b.id, role="user", content="same question"))
        await db.commit()

        count_a = len(
            (await db.execute(select(Message).where(Message.session_id == a.id))).scalars().all()
        )
        count_b = len(
            (await db.execute(select(Message).where(Message.session_id == b.id))).scalars().all()
        )
        assert count_a == 1
        assert count_b == 1


class TestNotFoundHandling:
    async def test_random_session_id_has_no_rows(self, db):
        random_id = uuid.uuid4()
        result = (
            await db.execute(select(Session).where(Session.id == random_id))
        ).scalar_one_or_none()
        assert result is None
