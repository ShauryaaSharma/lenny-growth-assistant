# 15 — The one error that most needs a request id was returned without one

**Context:** reading the body of the 500 from entry 14, not just its status.

## Symptom

```json
{"error": {"code": "internal_error",
  "message": "An unexpected error occurred.",
  "hint": "Check the server logs for the matching request_id.",
  "request_id": "-"}}
```

The hint tells the reader to find the matching `request_id` in the logs, and
the response gives them `-`. The server's own `unhandled_exception` log line
also said `request_id=-`, so there was nothing to match on either side — on
exactly the class of error (an unexpected crash) the request id exists for.
The 500 response also had no `X-Request-ID` header, which every other
response carries.

## Root cause

The request middleware sets the request id in a context variable and resets
it in a `finally` block. For an unhandled exception, that `finally` runs as
the exception propagates out of the middleware — before Starlette's outermost
`ServerErrorMiddleware` calls the `Exception` handler that builds the 500.
By then the context variable is back to its default, `-`. Handled errors
(404, 422, LLM errors) never hit this because their handlers run inside the
middleware.

## Fix

The middleware also stores the id on `request.state`, which outlives the
context variable. The `Exception` handler restores it into the context
variable for its log line, builds the envelope, then sets the
`X-Request-ID` header on the response.

## Verified

Against the running API, the next 500 Schemathesis provoked (entry 16)
carried `"request_id": "5a4971f48f96"`. Pinned by
`tests/test_contract_regressions.py`, which forces an unhandled error and
checks that the body's request id matches the response header and the id the
client sent.

Not found by a contract check: Schemathesis passed this response, because the
envelope shape was right. It surfaced only because the failing output was
read rather than just counted.
