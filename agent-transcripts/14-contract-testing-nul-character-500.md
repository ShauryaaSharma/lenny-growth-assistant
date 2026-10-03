# 14 — Contract testing found a 500 behind a single NUL character

**Context:** running Schemathesis against `/openapi.json` for the first time,
generating requests for every non-LLM endpoint and checking each response
against what the schema documents.

## Symptom

```
POST /api/sessions
- Server error
- Undocumented HTTP status code
    Received: 500
    Documented: 201, 422
Reproduce with:
    curl -X POST -H 'Content-Type: application/json' -d '{"title": "\u0000"}' http://127.0.0.1:8000/api/sessions
```

A session titled with one NUL character crashed the request. Everything else
Schemathesis tried against that endpoint passed.

## Root cause

PostgreSQL `text` columns cannot store U+0000, and `jsonb` rejects it too.
Pydantic accepted the string — it is valid JSON and a valid Python `str` —
so it reached the database, asyncpg raised, and the global handler turned
that into a 500. The same path was open on every user-supplied string the
API stores or queries with: the chat message, the search query and guest
filter, and every key or value inside `user_metadata` (a `jsonb` column).

## Fix

`reject_nul()` in `app/schemas/api.py`, applied as a validator to
`SessionCreate.title`, `SessionCreate.user_metadata` (recursively, keys and
values), `ChatRequest.message`, `SearchRequest.query` and
`SearchRequest.guest`. The input is now a 422 with the usual envelope:

```json
{"error": {"code": "validation_error", ...,
  "fields": [{"loc": "body.title", "msg": "Value error, must not contain NUL (U+0000) characters"}]}}
```

Rejecting rather than silently stripping: dropping characters from what a
user typed would change their input without telling them.

## Verified

Before the fix all six cases (title, nested metadata value, metadata key,
message, query, guest) were accepted by the models; after it all six are
rejected and normal input is unchanged. Pinned by
`tests/test_contract_regressions.py`. The next Schemathesis run was clean on
this endpoint.
