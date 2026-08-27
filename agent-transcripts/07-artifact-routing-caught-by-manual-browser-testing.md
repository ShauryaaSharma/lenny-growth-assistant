# 07 — Artifact routing failure, caught only by testing in a real browser

**Context:** the system prompt's rule 8 ("call `create_artifact` when the user
asks for a document, report, one-pager, table, checklist, template, or web
page") had passed every unit test. It had never been exercised against the
live model through the actual UI. Per this project's own guidance ("start the
dev server and use the feature in a browser before reporting the task as
complete"), it was tested there before being marked done.

## What happened

With ingestion running in the background and the full stack up in Docker, the
suggested prompt **"Build me a one-page onboarding audit checklist"** was
clicked in the real browser.

The backend logs showed:

```
retrieval_complete   grounded=True  hits=8
tool_executed         tool=search_transcripts  ok=True
agent_turn_complete   artifacts=0  tools_used=['search_transcripts']
```

`artifacts=0`. The model had searched correctly, retrieved good evidence, and
then — instead of calling `create_artifact` — wrote the checklist directly into
the chat message as a numbered list of questions. No artifact chip appeared in
the UI; the "document" the user asked for was just prose in the conversation.

This is exactly the kind of gap unit tests with a scripted `FakeProvider`
cannot catch: the *scripted* tests assume the model calls the right tool and
verify the surrounding logic handles that correctly. They cannot tell you
whether a real 3B model, given the real system prompt, actually chooses to
call it. Only running it for real, in the browser, against the model this
project is required to demo on, surfaced this.

## The fix

This is structurally the same reliability problem the forced-retrieval nudge
already solves for search — "the system prompt says to do X" is not reliable
routing insurance on a small model — so the same pattern was applied:

1. Added `wants_artifact()` in `app/agent/prompts.py`: a deliberately
   conservative keyword match (`"checklist"`, `"one-pager"`, `"template"`,
   `"document"`, `"report"`, `"audit"`, ...) against the user's message.
2. Added `FORCE_ARTIFACT_NUDGE`, injected once if the user's message matched
   and the turn is about to end with zero artifacts created.
3. Wired it into `run_agent()` in `app/agent/runtime.py`, guarded so it only
   fires when the question was actually answerable (never forces an artifact
   out of an honest refusal — see the bug below).

## A second, more subtle bug found while wiring the fix in

While adding the new guard's condition — "only force an artifact if the answer
wasn't an ungrounded refusal" — inspection of the existing ungrounded-guard
code turned up a real, independent bug it was piggybacking on:

```python
if ctx.searched and not ctx.grounded and UNGROUNDED_GUARD not in messages[-1].content:
    ...
    ctx.grounded = True  # guard fires once; next answer is accepted
```

`ctx.grounded` was being reused to mean two different things: "retrieval
actually found relevant material" (its real, documented meaning, also read
directly into the API response's `grounded` field) and "the ungrounded guard
has already fired once, don't loop forever" (a bookkeeping detail that has
nothing to do with retrieval quality). The practical consequence: **after an
honest refusal — "Lenny's Podcast transcripts don't cover this topic" — the
API response reported `grounded: true`**, which is the opposite of what
actually happened and directly contradicts the PRD's own success metric ("0%
of out-of-domain questions answered without evidence... this metric must never
be improved by lowering the floor" — silently mislabeling a refusal as grounded
is exactly the kind of quiet metric inflation that guardrail exists to prevent).

**Fix:** introduced a separate local variable, `ungrounded_guard_fired`, purely
for loop bookkeeping, and left `ctx.grounded` as the single source of truth for
whether the answer is actually backed by evidence.

## Tests added, and what they lock in

Four new tests in `TestForcedArtifactCreation`
(`backend/tests/test_agent_routing.py`):

- A document request answered in prose is corrected by the nudge, and the
  resulting artifact is registered.
- The nudge fires **at most once** — a model that ignores it twice must not
  trap the user in an infinite loop.
- A document request the corpus cannot support stays a refusal; the nudge does
  not force an artifact out of an honest "I don't know."
- An ordinary question never triggers the nudge at all (no false positives).

Plus one regression test on the pre-existing bug: an ungrounded refusal must
report `result.grounded is False` to the API, not `True`.

All four new tests, plus the regression test, verified against the fake
provider first (fast, deterministic), then the fix was rebuilt into the Docker
image and **the exact same "Build me a one-page onboarding audit checklist"
request was re-run live in the browser against the real `llama3.2:3b` model**.
The logs this time:

```
tool_executed              tool=search_transcripts   ok=True
forcing_artifact_creation
tool_executed              tool=create_artifact      ok=True
agent_turn_complete        artifacts=1  tools_used=['search_transcripts', 'create_artifact']
```

And the artifact panel opened with a genuinely useful, grounded document (a
hiring checklist citing Adam Fishman's and Annie Pearl's specific advice on
evaluating PM candidates) rather than a chat message.

## Why this is worth reading

Two real defects here — one a missing behavior, one a mislabeled metric — were
both found by the same discipline: don't mark a feature done because its unit
tests pass; run it for real, against the actual required model, in the actual
UI, and read the actual logs. The fix for the first defect is what led directly
to discovering the second while writing its guard condition.
