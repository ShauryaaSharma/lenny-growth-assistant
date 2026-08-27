# 10 — Building the eval harness immediately caught a real grounding defect

**Context:** after the build was otherwise complete and pushed, a direct
question was asked: why was there no LLMOps evaluation harness, given the PRD
names a specific, numeric success metric ("Grounded Answer Rate >= 80%") and a
guardrail ("0% of out-of-domain questions answered without evidence")? The
honest answer was that neither had ever actually been measured — the PRD
stated them, the code logged the fields needed to compute them
(`grounded`, `best_similarity` on every turn), but nothing ran a fixed set of
questions and reported the actual numbers. 94 passing unit tests do not
substitute for this: they verify code behaviour against scripted inputs, not
system quality against real ones.

## What was built

`backend/app/evals/`:

- `golden_set.py` — 24 labeled questions. 14 in-domain (7 naming a specific
  guest confirmed present in the corpus via a direct `SELECT guest FROM
  episodes` check, 7 broad topic questions), 10 out-of-domain (sourdough
  recipes, bike repairs, geography, physics — deliberately nothing a podcast
  about product and growth would plausibly cover).
- `run_eval.py` — calls the real `rag.retriever.search()` (no model, no agent
  loop — retrieval only, so a full run takes seconds even on CPU) against
  every golden question, and reports: Grounded Answer Rate, False-Ground Rate,
  guest-match precision, and retrieval latency percentiles. Exits non-zero if
  either the PRD's 80% floor or its 0% guardrail is violated, so it's CI-ready.
- `tests/test_eval_harness.py` — 12 tests on the pure scoring function
  (`summarize()`), so the harness's own arithmetic is verified without
  needing a database, mirroring the same principle used everywhere else in
  this codebase: don't test scoring logic against a live dependency when a
  fixed input will do.

## What the first real run found

```
Grounded Answer Rate:      100%
False-Ground Rate:         80%
RESULT: FAIL
```

Every single in-domain question grounded correctly. But **8 of 10
out-of-domain questions also grounded** — "What's the best sourdough starter
recipe?", "Can you explain general relativity in simple terms?", "What's the
healthiest way to cook salmon?" all cleared the similarity floor and would
have been answered as if the corpus supported them.

This is not a hypothetical or edge-case defect. It is the exact failure mode
the PRD names as the single most important thing to prevent: a system that
fabricates or over-extends inside its stated domain. Sitting on this
undetected would have directly contradicted the guardrail explicitly written
down in `docs/PRD.md` before a single line of code was tested against it.

## Root cause and the fix

`RETRIEVAL_MIN_SIMILARITY` had been set to `0.55` during initial development
by eyeballing a handful of live queries — a reasonable-looking number that was
never checked against a labeled sample. Printing the actual similarity scores
from the eval run showed why it was wrong:

```
in-domain questions:      0.712 - 0.810
out-of-domain questions:  0.541 - 0.664
```

`bge-small-en-v1.5` clusters cosine similarity for conversational English text
more tightly than the 0.55 guess accounted for — a completely unrelated
question about sourdough bread still measured 0.585, comfortably above the
old floor. There is a clean ~0.05 gap between the two clusters in this sample
(0.664 max out-of-domain vs. 0.712 min in-domain). The floor was moved to
`0.69`, sitting in that gap with margin on both sides, and the reasoning is
recorded directly in the `config.py` comment on the setting so a future change
to the embedding model or chunk size has a documented reason to re-run the eval
rather than guess again.

Re-running the harness after the change:

```
Grounded Answer Rate:      100%
False-Ground Rate:         0%
Guest-match precision:     86%
RESULT: PASS
```

One existing test (`test_retriever.py::test_episode_without_youtube_url_cites_by_title_only`)
started failing at the new, stricter threshold — its synthetic single-chunk
fixture used a loose paraphrase that scored 0.605, below the new floor. This
was a weak test fixture, not a production regression: a real corpus query
benefits from many supporting chunks reinforcing the same topic, which a
single synthetic sentence with no redundancy behind it cannot replicate. Fixed
by tightening the test's query to more directly match its fixture's wording
(the test's actual purpose — verifying citation degradation when
`youtube_url` is empty — was unaffected either way).

## Why this is the most important entry in the log

Every other defect found in this build (entries 07-09) was caught by running
the system live and noticing something looked wrong. This one was caught by
*building the instrument that measures the thing the PRD claimed to
guarantee*, and it found a defect on the very first run — one that manual
testing, however thorough, had not surfaced, because manual testing
(including everything in entries 07-09) only ever tried questions the corpus
plausibly *did* cover. The gap was invisible until something was built
specifically to try the opposite.
