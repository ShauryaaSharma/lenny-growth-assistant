# 08 — Two more defects found by running the Ship 30 essay live, for the first time

**Context:** after fixing the artifact-routing gap (entry 07), the Ship 30
essay skill was tested end-to-end for the first time against the real running
stack — clicking "Write a Ship 30 essay about retention as a growth lever" in
the real browser, against the real `llama3.2:3b` model, with the real
ingested corpus. Like entry 07, this had passed every unit test up to this
point (the rubric-check unit tests didn't exist yet — see below) but had never
been run for real.

## Defect 1 — conflicting tool instructions in one turn

The backend logs showed:

```
forcing_retrieval                       (model tried to answer without searching)
retrieval_complete   grounded=True  hits=8
tool_executed         tool=search_transcripts
agent_turn_complete   tools_used=['write_ship30_essay', 'search_transcripts']
```

The model, after being nudged to search, responded with **two tool calls in
the same message**: both `write_ship30_essay` and `search_transcripts`. The
essay was generated correctly and registered as an artifact. But the chat
reply shown to the user was:

> "Based on the search results, here is a one-page hiring checklist for a
> first PM hire: What is the onboarding process for new customers? [1] ..."

This is completely unrelated to the essay that was actually created (titled
"The Power of Incentives and Learnings in Growth Teams") — it reads like a
continuation of a *previous, unrelated* request in the same session. The
`write_ship30_essay` tool result carries the instruction "describe it in 2-3
sentences, do not reproduce it"; the `search_transcripts` tool result (called
in the same turn) carries a *different* instruction, "answer using these
excerpts, cite as [1], [2]." Both instructions were present in the model's
context at once, and it followed the wrong one.

**Fix:** `ARTIFACT_JUST_CREATED_REMINDER` (`app/agent/prompts.py`) is now
appended as the literal last message in the conversation immediately after any
tool call that creates an artifact — regardless of what other tools ran in the
same batch, and regardless of what those other tools' own result instructions
said. Small models weigh recent context heavily; putting the correct
instruction last, after everything else, is what makes it win. Guarded to fire
at most once per turn even if multiple artifacts are created.

Three new tests in `TestArtifactCreatedReminder`
(`backend/tests/test_agent_routing.py`) lock this in, including one that
reproduces the exact multi-tool-call scenario directly (constructing an
`LLMResponse` with both `create_artifact` and `search_transcripts` tool calls
in one message) rather than only testing the single-tool-call happy path.

## Defect 2 — a literal "[n]" leaked into the essay

Reading the rendered essay in the browser, the inline citations were literally
the three characters `[n]`, not real numbers:

> "As Archie Abrams notes, 'Incentives, what a power, what a lever.' **[n]**"

Tracing this back: `_outline_prompt` in `app/skills/ship30/skill.py` instructed
the model to "list the evidence numbers **[n]** that support it," using `[n]`
as meta-notation meaning "put a number here" — standard convention in
documentation, but a 3B model has no reliable way to distinguish that from a
literal instruction to write the two characters `n` inside brackets. The
`_revision_prompt`'s citation-failure message had the identical bug ("cite...
inline as **[n]**"). Since both the initial generation *and* its one revision
pass used the same flawed instruction, the defect survived the pipeline's own
quality gate — the rubric's citation regex (`\[(\d{1,2})\]`) correctly does
not match `[n]` (`n` is not a digit), so `has_three_citations` correctly
failed and triggered a revision, but the revision instruction told the model
to do the exact same wrong thing again.

**Fix, two parts:**

1. Reworded both prompts to give a concrete example ("write `[1]` or `[3]`,
   using the actual number from the evidence list — never write the literal
   characters `[n]`") instead of symbolic placeholder notation.
2. Added a **named, distinct rubric check** — `no_literal_placeholder_citations`
   — rather than leaving this folded invisibly into the generic citation-count
   check. This matters for two reasons: it makes the specific failure visible
   in the rubric report rather than looking like "just not enough citations,"
   and it lets the revision instruction name the exact defect
   ("the essay contains the literal characters `[n]`... replace every `[n]`
   with the actual number") instead of a generic count-based nudge that had
   already been shown not to fix this particular failure mode.

`backend/tests/test_ship30_skill.py` was created (previously, this skill's
pure rubric/revision logic had **no direct unit tests at all** — only indirect
coverage through agent-routing tests that mock the essay tool entirely and
never exercise `check_rubric` or `_revision_prompt`). It includes a test that
reproduces this exact defect (an essay with `[n]` substituted for every real
citation) and asserts it is caught by the new named check, plus a test that
the revision instruction explicitly names the literal-placeholder defect.

## Why both entries matter together

Neither of these two defects would have been caught by the existing test
suite, because the existing suite mocked the essay generation entirely in
agent-routing tests, and had no dedicated tests for the skill's internal rubric
logic at all. Both were found the same way as entry 07: by actually running
the feature, against the actual required model, through the actual UI, and
reading the actual output — not by inspecting code or trusting that "the
rubric checks for citations" meant citations would be correct.
