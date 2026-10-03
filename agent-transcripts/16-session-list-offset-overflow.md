# 16 — An unbounded `offset` overflowed Postgres and became a 500

**Context:** the second Schemathesis run, after fixing entry 14, at 100
generated cases per operation.

## Symptom

```
GET /api/sessions
- Server error
- Undocumented HTTP status code
    Received: 500
    Documented: 200, 422
Reproduce with:
    curl -X GET 'http://127.0.0.1:8000/api/sessions?offset=9223372036854775808'
```

## Root cause

`offset` was declared `Query(default=0, ge=0)` — a lower bound and no upper
one. Postgres `OFFSET` is a `bigint`, whose maximum is 2⁶³ − 1
(9,223,372,036,854,775,807). Schemathesis sent exactly one more; asyncpg
could not encode it as a 64-bit integer and raised, and that became a 500.
`limit` was not affected: it already had `le=200`.

## Fix

`offset: int = Query(default=0, ge=0, le=PG_BIGINT_MAX)` in
`app/api/routes_sessions.py`, with `PG_BIGINT_MAX = 2**63 - 1`. The bound is
the one Postgres itself imposes, not an arbitrary product limit, so no value
that used to work is now refused.

## Verified

```
offset=9223372036854775807 -> HTTP 200
offset=9223372036854775808 -> HTTP 422
```

The next Schemathesis run was clean: 11 operations, 803 generated requests,
every check passed. Pinned by `tests/test_contract_regressions.py` at both
sides of the boundary.
