# 01 — Chunking defects found by testing early

**Context:** after writing `app/rag/chunking.py` against the transcript format,
instead of writing unit tests against a hand-written sample first, the agent
ran the parser directly against all 303 real transcript files from the cloned
corpus, and printed aggregate statistics.

## What the test run showed

```
episodes: 303
parse failures: 4
   daniel-lereya/transcript.md: missing video_id
   nickey-skarstad/transcript.md: missing video_id
   peter-deng/transcript.md: missing video_id
   teaser_2021/transcript.md: missing video_id
chunks: 8514  sponsor: 482 (5.7%)
tokens/chunk median 716.0  p95 1024  max 4238
```

## Defect 1 — chunk size exceeded the embedding model's context window

The initial chunk target was 800 tokens with 100 overlap, chosen without
checking it against the specific embedding model that would encode those
chunks. The p95/max figures above (1024 / 4238 tokens) are well past
`bge-small-en-v1.5`'s 512-token limit. An embedding model does not error on an
over-length input — it silently truncates, so the failure mode here is not a
crash, it is *unretrievable content with no error signal at all*.

**Fix:** chunk target reduced to 400 tokens / 80 overlap, and a
`_split_long_turn` function was added to break a single monologue longer than
the budget on sentence boundaries (guests occasionally talk for several
thousand tokens uninterrupted; without this a single turn could still exceed
the window even with a smaller target). Re-running the same test:

```
episodes 303  failures 0
chunks 17785  sponsor 809 (4.5%)
median 364  p95 451  max 513
over 512: 1
```

Reduced from "1024/4238 tokens, unbounded" to "451/513 tokens, one edge case,"
verified by rerunning the exact same measurement rather than assuming the fix
worked.

## Defect 2 — four episodes were being silently dropped

The frontmatter parser initially raised `ValueError` on a missing `video_id`,
which is a reasonable validation rule in isolation but turned out to discard
four real, fully-transcribed episodes whose upstream metadata ships empty
(`video_id: ''`). This was only visible because the test ran against the
*actual* 303-file corpus rather than a synthetic three-file fixture that would
never have exercised this path.

**Fix:** fall back to a stable slug-derived id (`slug:{folder-name}`) when
`video_id` is empty, and leave `youtube_url` empty rather than fabricating one
— such episodes cite by title and guest only, with no deep-linkable timestamp,
which is the honest representation of what's actually knowable about them.

## Defect 3 — Ollama's IPv4-only bind

Not caught by this specific test run, but discovered in the same de-risking
pass: `curl http://localhost:11434/api/tags` failed while
`curl http://127.0.0.1:11434/api/tags` succeeded, because `localhost` resolved
to `::1` and Ollama does not listen there. Recorded directly in the README's
troubleshooting section rather than left for an evaluator to hit blind.

## Why this is the entry worth reading

None of these three defects would have raised an exception anywhere else in
the system. All three would have manifested only as "the answers are worse
than they should be" or "I can't reach the model" — the hardest class of bug to
trace backward from a symptom to a root cause. Running the real chunker
against the real corpus before building the retrieval layer on top of it is
what converted them from a production-debugging problem into a five-minute fix.
