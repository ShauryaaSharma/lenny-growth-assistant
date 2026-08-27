# 06 — A test caught a real SQLAlchemy async gotcha, and the first fix for it was also wrong

**Context:** writing `tests/test_sessions.py` to verify that deleting a message
does not cascade-delete its associated artifact (the schema uses
`ON DELETE SET NULL` on `Artifact.message_id`, not `CASCADE`, specifically so a
document the user is looking at survives if the triggering message is later
pruned).

## Failure 1

```python
await db.delete(m)
await db.commit()
refreshed = await db.get(Artifact, art.id)
assert refreshed.message_id is None
# AssertionError: assert UUID('4fbbeeb0-...') is None
```

The database had, in fact, correctly applied `SET NULL` — this was confirmed
separately with a raw `psql` query. The test was failing on stale application
state, not a schema bug: SQLAlchemy's session-level identity map was still
holding the pre-delete Python object, and `db.get()` returns the cached object
for a primary key already present in the session rather than re-querying,
unless told the cached copy is invalid.

## Failure 2 — the first attempted fix

```python
db.expire_all()
refreshed = await db.get(Artifact, art.id)
# sqlalchemy.exc.MissingGreenlet: greenlet_spawn has not been called;
# can't call await_only() here. Was IO attempted in an unexpected place?
```

`expire_all()` marks attributes as stale but defers the actual reload to the
next attribute access — which is normally fine, but async SQLAlchemy cannot
perform that implicit reload outside of an already-awaited context, and
`db.get()` on an object whose PK is already in the identity map doesn't trigger
the kind of IO this error message is asking for.

## Actual fix

```python
await db.refresh(art)
assert art.message_id is None
```

An explicit, awaited `refresh()` on the specific object is the correct pattern
in async SQLAlchemy for "I know the database changed this row out from under
my session's cache, get me the current values." Verified by rerunning: passed
cleanly, and the rest of the suite (61 other tests) still passed on the same
run, confirming this was an isolated fix rather than a change that broke
something else.

## Why this belongs in the log

This is a bug that only ever existed in the test, never in application code —
the real API layer never deletes an individual message independent of its
whole session, so this staleness pattern has no live production path today.
It's included anyway because it is a genuine "first fix was also wrong, second
fix was verified before being trusted" sequence, and because it is exactly the
kind of async-ORM footgun worth documenting for whoever extends this codebase
next and does add a path that deletes messages individually.
