# 11 — Building an agent harness (not just an eval harness) found three more real bugs, including two live hallucinations

**Context:** asked directly "do we have any agent harness?" after the retrieval
eval harness (entry 10) shipped. The honest answer was no: `app/evals/`
tested retrieval only, never the real `run_agent()` tool-calling loop against
a real model. `tests/test_agent_routing.py` tested the guard *logic* against
a scripted `FakeProvider` that only ever behaves the way its script says --
useful, but it can only catch bugs someone already imagined when writing the
script. Built `app/evals/agent_scenarios.py` + `run_agent_eval.py`: 8 golden
conversations run through the real agent loop, real tool registry, real
guards, against the real required local model, scored on tool-call
correctness, refusal correctness, artifact correctness, and redundant-call
detection.

## What the first live run found

```
[PASS] trivial          greeting_no_tools           <- fine
[PASS] trivial          thanks_no_tools             <- fine
[PASS] grounded_qa      grounded_question_onboarding <- fine
[FAIL] refusal          out_of_domain_refusal
[FAIL] artifact         document_request_creates_artifact
[FAIL] artifact_refusal ungrounded_document_request_stays_refused
[PASS] multi_turn       two_turn_followup_context
RESULT: FAIL
```

Two of these were genuine, serious defects -- live hallucinations, not
near-misses:

- **"What's the best sourdough starter recipe?"** -- the model never called
  `search_transcripts` at all. It called `create_artifact` directly and
  rendered a fully fabricated sourdough recipe as a legitimate-looking
  document.
- **"Give me a one-page sourdough starter checklist"** -- same pattern: no
  search, straight to `create_artifact`, a fabricated checklist rendered as
  an artifact.

## Root cause

The forced-retrieval guard (`if needs_grounding and not ctx.searched and not
nudged`) only fires when the model gives a *bare text answer* without having
searched. It has no opinion about what happens if the model instead calls
*some other tool* first. `create_artifact` was freely callable regardless of
whether anything had been searched or grounded -- the tool trusted whatever
content the model supplied unconditionally. A model that wants to skip
retrieval does not need to answer in prose to do it; it only needs to call a
different tool. This is exactly the kind of failure mode a scripted
`FakeProvider` test cannot produce, because the fake only ever does what its
author wrote into the script, and no one had scripted "the model calls the
wrong tool as an escape hatch" -- a real model did it unprompted, twice, on
the very first live run.

The third failure, `document_request_creates_artifact`
(`tools=['create_artifact', 'search_transcripts', 'create_artifact']`), was a
different real gap: nothing capped how many artifacts could be created in one
turn. The model created one artifact correctly, then searched again, then
created a *second*, unwanted artifact in the same turn.

## The fix

Three changes to `app/agent/runtime.py`:

1. **No tools on trivial messages.** A side discovery while investigating:
   "hi" was calling `search_transcripts` with a hallucinated, unrelated query
   ("how to build trust and grow as a product leader"), costing ~150s for a
   greeting. `needs_grounding` now also gates whether tools are offered at
   all (`tools=None` for a trivial message), not just whether a bare answer
   gets force-nudged.
2. **`create_artifact` is intercepted, not trusted, before it runs.** If the
   turn needed grounding and hasn't searched yet, the call never executes --
   it is replaced with a synthetic tool error instructing the model to search
   first, recorded in the trace as a blocked attempt (`ok: False`).
3. **A second content-creating call in the same turn is blocked outright.**
   Once one artifact exists, any further `create_artifact` or
   `write_ship30_essay` call in that turn is intercepted the same way.

## The harness itself had two bugs, found while re-verifying the fix

Re-running after the fix, three scenarios still showed FAIL -- but reading
the actual replies showed the *agent* was now correct in two of the three
cases; the *harness's own scoring* was wrong:

- The refusal-language check missed the plural phrasing a real reply used
  ("Lenny's Podcast transcripts **do not cover** the topic..." vs. the
  marker list's singular-only "doesn't cover"/"does not cover"). Fixed by
  adding plural forms to `REFUSAL_MARKERS`.
- The redundant-call check counted a *blocked* `create_artifact` attempt
  (`ok: False`) the same as a successfully executed one, so the runtime's own
  new guard doing its job correctly ("attempted twice, executed once") was
  scored as a defect. Fixed by filtering `score_turn()`'s checks to executed
  (`ok=True`) calls only -- a blocked attempt stays visible in the trace but
  no longer fails the check it exists to satisfy.

Also fixed, found while running the harness under concurrent ingestion load:
an unhandled `LLMTimeoutError` on one scenario was crashing the entire run
and losing every other scenario's result. `run_scenario()` now catches
provider errors per-turn and records them as a failed check, same as a wrong
routing decision, rather than propagating.

Final clean run (no ingestion contention): **8/8 scenarios, 100% across every
category**, including the essay scenario completing in one generation with no
redundant call -- the direct regression guard for the entry-09 defect.

## Why this entry matters most

Entries 07-09 were each found by me happening to try the right question in
the browser. This is the first defect-finding mechanism in the whole build
that is automated, repeatable, and would keep working after I stop manually
testing. It also demonstrates something entries 07-10 didn't: the harness
that finds real bugs can itself have real bugs, and the fix isn't "trust the
harness less," it's the same discipline applied one level up -- read what the
model actually said before accepting a FAIL, and verify a scoring fix the
same way any other fix gets verified.
