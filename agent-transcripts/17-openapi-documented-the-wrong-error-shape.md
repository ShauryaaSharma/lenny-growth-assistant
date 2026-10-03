# 17 — The OpenAPI schema documented an error shape the API never returns

**Context:** after the contract run went clean, checking *why* the response
schema check had passed on 422s at all.

## Symptom

Every 422 the API returns looks like this:

```json
{"error": {"code": "validation_error", "message": "...", "hint": "...",
  "request_id": "ab3c0d357896",
  "fields": [{"loc": "body.query", "msg": "String should have at least 1 character"}]}}
```

But `/openapi.json` described every 422 as FastAPI's default
`HTTPValidationError` — `{"detail": [{"loc": [...], "msg": ..., "type": ...}]}`
— and documented no 404 on any route, though five of them return one, and no
model-provider error (502/503/504) on chat.

## Root cause

The API replaces FastAPI's validation response with its own envelope (the
`RequestValidationError` handler in `app/main.py`) but never told the schema.
Schemathesis did not flag it because FastAPI's default schema does not mark
`detail` as required, so `{"error": ...}` validated against it. Anyone
generating a client from `/openapi.json` would have been told to read a
`detail` array that never arrives.

## Fix

`app/api/errors.py` declares the real responses — `INVALID` (422,
`ValidationErrorResponse`), `NOT_FOUND` (404, `ErrorResponse`) and
`MODEL_PROVIDER` (502/503/504) — and each route lists the ones it can
actually return. Declaring 422 on a route replaces FastAPI's default for that
route; `HTTPValidationError` is no longer in the schema at all.

## Verified

The Schemathesis run against the corrected schema passed 803/803 — now
checking every 422 body against the real envelope. To confirm the check has
teeth, `ValidationErrorDetail` was temporarily given a required field the API
never sends; every 422 then failed with `'never_sent' is a required
property`, and the change was reverted. Pinned by an OpenAPI test in
`tests/test_contract_regressions.py`.

This one was found by reading the schema, not by a failing check — a clean
contract run is only as strict as the schema it checks against.
