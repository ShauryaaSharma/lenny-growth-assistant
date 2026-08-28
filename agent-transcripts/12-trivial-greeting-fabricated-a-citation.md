# 12 — A bare "Hey" fabricated a quoted citation, with an invented guest name

**Context:** manually smoke-testing the live app in the browser (not a
scripted test) while preparing to record the demo video. Sent the single
word "Hey" as a fresh session's first message.

## What happened

```
Hey
[1] "I think the biggest mistake I made was assuming I knew what I was
doing." - Lenny Russell

llama3.2:3b · 16.2s
```

Two problems in one reply:

- A fabricated quote, presented with a `[1]` citation marker exactly like a
  real grounded answer -- but no `search_transcripts` call happened at all
  (confirmed via the trace store, `GET /api/sessions/{id}/trace`: zero spans
  for that turn beyond the LLM call itself).
- An invented guest name, "Lenny Russell." Lenny's Podcast is hosted by Lenny
  Rachitsky; no guest by that name exists in the corpus. The model
  free-associated a plausible-sounding "Lenny [surname]" from the word
  "Lenny" in its own persona description.

## Root cause

`is_trivial()` worked correctly -- "hey" is 3 characters, under the
`len(normalized) <= 3` threshold, so `needs_grounding` was `False` and
`tool_specs` was `None`. The routing guard did its job: no tool was ever
offered or called.

The gap was one level up, in `SYSTEM_PROMPT` itself
(`backend/app/agent/prompts.py`). Its rules cover what to do for a
*substantive* question (rule 1: search first) and hammer the citation format
relentlessly (rule 2: "Cite them inline as [1], [2]"; rule 3: "Attribute
ideas to the guest who said them"). Nothing told the model what to do when
neither applies. Left to fill that gap on its own, a 3B model pattern-matched
its own citation habit and produced a citation-shaped answer anyway --
inventing both the quote and the source to fit the shape it had been told to
always use. This is the same class of bug as entry 11's live hallucinations
(the model finding an unguarded path to free-generate instead of grounding),
just one level earlier: the *routing* guard was never the gap here, the
*prompt* was silent on what "no grounding available" should sound like.

## The fix

One new rule added to `SYSTEM_PROMPT`:

```
10. For greetings, thanks, or small talk with no product/growth question in
it, reply briefly and naturally in your own words. Do not search, cite
[1]/[2], quote anyone, or name a guest -- there is nothing to ground yet.
```

Re-tested live after the fix: the same "Hey" now gets `"How's it going?"` --
`grounded: false`, `tool_calls: []`, no citation markers, no invented name.

Also added a regression guard the existing `greeting_no_tools` /
`thanks_no_tools` agent-harness scenarios could not have caught: they only
asserted `forbidden_tools` (routing), and this bug involved zero tool calls
by construction. A new `forbidden_reply_patterns` field on `ScenarioTurn`
(`app/evals/agent_scenarios.py`, wired into `score_turn()` in
`run_agent_eval.py`) checks the *reply text itself* for citation markers, and
is now set on both trivial scenarios. Two matching unit tests were added to
`tests/test_agent_eval_harness.py`. This is the kind of check the harness
should have had from the start: routing correctness and reply-content
correctness are different failure surfaces, and entry 11's scenarios only
ever measured the first one.
