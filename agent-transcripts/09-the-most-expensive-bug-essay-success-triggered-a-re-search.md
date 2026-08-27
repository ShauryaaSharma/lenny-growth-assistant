# 09 — The most expensive bug in the build: success triggered a retry

**Context:** immediately after fixing the two defects in entry 08 (conflicting
tool instructions, literal `"[n]"`), the fix was re-verified live by running
the exact same Ship 30 essay request again. It appeared to hang: a `curl`
request with a 700-second timeout gave up before the server responded.

## What was actually happening

Checking the backend logs (not just assuming the timeout meant "broken")
showed the true sequence:

```
tool_executed   tool=write_ship30_essay   ok=False   latency_ms=297235   (Ollama LLMTimeoutError)
forcing_retrieval
retrieval_complete   grounded=True
tool_executed   tool=search_transcripts
retrieval_complete   grounded=True   (a SECOND, redundant search)
tool_executed   tool=create_artifact   ok=False   (empty content -- correctly rejected)
agent_iteration_limit   iterations=5
```

The essay generation had failed once on an internal Ollama timeout (itself
caused by the concurrently-running ingestion job saturating the CPU — a
separate, expected form of contention, already documented). But what happened
*after* that failure was the real defect: the model searched again, and then
the loop hit its iteration cap without ever completing.

To isolate the actual bug from the CPU-contention noise, the background
ingestion process was killed cleanly (found via `docker exec ... /proc`
inspection when `pkill`/`ps` weren't available in the slim container image)
and the exact same request was re-run with the full 16 cores available. It
**still took nearly 6 minutes and — before this fix — the logs showed the
essay tool succeeding, followed immediately by a second full essay
generation.** This was not a contention artifact. It was a real bug that
would double the cost of every successful essay, unconditionally, forever.

## Root cause

`_tool_write_ship30_essay` (`app/agent/tools.py`) sets `ctx.grounded = True` on
success, but never set `ctx.searched = True`. The agent loop's forced-retrieval
guard checks `needs_grounding and not ctx.searched` — it has no way to know
that the essay tool performs its own internal, wider retrieval (`top_k=14`,
inside `write_ship30_essay()` in the skill module) entirely separately from
the top-level `search_transcripts` tool. After a successful essay, the loop
saw `ctx.searched == False`, concluded the model had never searched, and
injected `FORCE_SEARCH_NUDGE` — telling a model that had just spent several
minutes producing a correct, grounded, 984-word essay that it needed to search
before answering. The model's response to that confusing instruction was to
search again, get confused about what task it was now doing, and attempt the
entire essay pipeline a second time.

## The fix, and proving it actually mattered

```python
result = await write_ship30_essay(ctx.db, topic)
ctx.searched = True   # set regardless of outcome -- see comment in the code
if not result["ok"]:
    return {"ok": False, "message": result["message"]}
ctx.grounded = True
```

Before writing the fix into a test, the bug was reproduced and the fix was
temporarily reverted (`ctx.searched = True` replaced with `pass`) to confirm
the new tests actually fail without it — a deliberate check against writing a
test that would pass regardless of whether the bug existed. **Six tests
failed** with the revert in place, including a new end-to-end test built
directly from the observed tool-call sequence
(`TestEssayCompletionDoesNotTriggerSpuriousResearch`), which asserts the exact
property that was broken: exactly one `write_ship30_essay` call, and no
`FORCE_SEARCH_NUDGE` in the final message list. The fix was then restored and
all tests re-confirmed passing (94 total).

Re-verified live a third time, again with ingestion paused for a clean signal:
one essay generation, ~6 minutes (all CPU-bound model inference, no
contention), a chat reply that correctly and coherently describes the actual
essay produced, real numbered citations throughout the essay body, and exactly
one artifact registered.

## Why this is the most important entry in the log

Every other defect found live in this session (entries 07 and 08) was a
correctness problem: the wrong thing happened once. This one was a *cost*
problem with no natural ceiling — every successful essay generation, forever,
would have silently cost roughly double what it should have, on the single
most expensive operation in the whole system (a multi-minute, multi-call
pipeline on the required local model). It is also the clearest example of why
"the tests pass" was never treated as equivalent to "the feature works": the
existing test suite (94 tests, all green) had no way to catch this, because no
test exercised the interaction between the essay tool's success path and the
top-level loop's retrieval bookkeeping — a gap that only running the real
pipeline, against the real model, twice, revealed.
