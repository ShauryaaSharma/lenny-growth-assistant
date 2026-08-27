# 03 — FastAPI crashed at import time on the DELETE route, twice

**Context:** first real container boot of the backend, after the image finally
built successfully.

## Failure 1

```
AssertionError: Status code 204 must not have a response body
  at routes_sessions.py:147, @router.delete("/{session_id}", status_code=204)
```

`delete_session` was declared with `-> None` and no explicit response class.
FastAPI infers a JSON response model from the return type annotation by
default, and a 204 route is asserted at import time to be incompatible with
having any response model at all.

**First fix attempted:** added `response_class=Response`. Rebuilt.

## Failure 2 — the same assertion, from the same line

```
AssertionError: Status code 204 must not have a response body
```

Identical error, after a fix that should have addressed it. This is the
interesting part: `response_class=Response` alone was not sufficient because
the file has `from __future__ import annotations` at the top, which makes
`-> None` a *string* ("None") at the point FastAPI inspects it, and FastAPI
still resolved that string into a real response model to validate against the
declared response class — reintroducing the same conflict.

**Actual fix:** two changes together —

```python
@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,   # explicit override; the annotation alone wasn't trusted
)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Response:
    ...
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

Setting `response_model=None` explicitly, and changing the function to
actually return a `Response` object rather than `None`, resolved it. Verified
by rebuilding and tailing the container logs until `Application startup
complete` appeared with no traceback — not by assuming the second fix worked
because it looked more correct than the first.

## Why this is worth recording

The first fix was a reasonable, plausible-looking change that did not actually
address the root cause (`from __future__ import annotations` interacting with
FastAPI's runtime type inspection). Shipping a fix without rebuilding and
re-checking the logs would have left this broken silently. The general lesson
applied for the rest of the session: after any fix to a startup-time crash,
rebuild and read the actual log output before moving on, rather than trusting
that the diagnosis was correct.
