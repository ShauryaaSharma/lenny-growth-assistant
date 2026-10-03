# 18 — A tenth of the corpus had no chunks, and another tenth was mislabelled

**Context:** the new knowledge-base checks (`python -m app.validation`) were
wired into a load-test script that ingests a 40-episode subset before
measuring anything. Their first run against real data, not the 3-episode CI
fixture, failed before a single request was sent:

```
FAIL  episode count matches the corpus       38 episodes, expected 40
FAIL  every episode has chunks               4 episodes have no chunks
                                             e.g. EW6K8ZOWoIs
                                             e.g. gCEaUfZUuI0
                                             e.g. J9UWaltU-7Q
                                             e.g. uMhBej6-Ey4
```

The ingestion run for the same data had reported `status: ok`, 40 seen,
40 ingested, 0 skipped.

## Defect 1 — 30 of 303 transcripts produced no chunks

**Symptom.** Four episodes existed, with title, guest and date, and no chunks,
so nothing in them could ever be retrieved.

**Root cause.** The parser read exactly one layout, `Speaker (HH:MM:SS):`.
Run over all 303 files, 30 matched no turn at all and so produced no chunks,
without an error, which is why nothing reported them. They use three other
layouts:

- `Speaker (MM:SS):`, with no hours field, in most of the 30;
- `[HH:MM:SS] Speaker: text` on one line (1 file);
- `Speaker:` with no timestamp at all (1 file).

This had been hiding since day zero. Transcript 01 measured the parser against
the whole corpus ("episodes 303, failures 0, chunks 17785"), but "failures"
counted parse *exceptions*, and these files parse without raising: they just
yield nothing. 17,785 is also exactly what the old parser produces today.

**A second defect in the same place.** In 242 of the files that did parse, a
bare `(00:01:21):` line, meaning the same speaker carrying on, didn't match the
header pattern either, so it was kept as *text*: 16,492 timestamp lines sat
inside turns, were embedded, were full-text indexed, and appeared in the
passages quoted back to the model and the user.

**Fix.** `TURN_RE` accepts `MM:SS` and a bare timestamp. The two rarer layouts
are tried only when a file has no timestamped turn, and they only accept a
name-like speaker label (one to five capitalised words), so a sentence ending in
a colon isn't taken for a speaker. Untimed turns have no start time, which the
retriever and UI already handled: the citation links to the video without a
`t=` and shows no timestamp.

**How it was checked against the whole corpus**, before and after:

| | Before | After |
|---|---|---|
| Files with no chunks | 30 | 0 |
| Chunks | 17,785 | 18,987 |
| Sponsor chunks | 809 (4.5%) | 877 (4.6%) |
| Tokens per chunk, median / p95 / max | 364 / 451 / 513 | 362 / 449 / 512 |
| Chunks over the model's 512-token window | 1 | 0 |

And for the 273 files that already parsed: the same words in the same order in
all 273, once the leaked timestamp lines are set aside.

## Defect 1b — my first fix un-flagged 212 paragraphs of ads

The first version of the fix made a bare timestamp start a *new turn* for the
same speaker, which seemed more precise, since each paragraph would get its own
timestamp. A diff of sponsor flags across the corpus said otherwise: 1,065
paragraphs lost their sponsor flag, and 212 of them were still plainly ad copy
("Visit DX's website at getdx.com/lenny"). Ad reads run over several
paragraphs separated by bare timestamps. `_flag_sponsors` judges per turn and
only sweeps the next turn along if it has a call-to-action marker of its own,
so splitting the read into turns left its middle unflagged and retrievable.

**Fix.** A bare timestamp is dropped and the turn goes on. Sponsor flags are now
identical to before: 0 paragraphs changed across the 273 files. A regression
test pins it (`test_a_multi_paragraph_ad_read_stays_one_sponsor_turn`).

**Left as it was, and why.** Most of those 1,065 paragraphs were *not* ads.
They were Lenny's welcome or the guest's first answer, flagged because they
share a turn with the ad read before them. That over-flagging predates this
change, and fixing it means sponsor detection per paragraph rather than per
turn, which is a retrieval change to measure on its own, not to slip into a
parser fix.

## Defect 2 — 31 episodes silently overwrote 31 others

**Symptom.** 40 transcripts, 38 episodes, and no duplicate video ids in the
table.

**Root cause.** Upstream, 31 video ids each appear in two folders: 62 of the
303. In every pair, the title, URL, id and date of one episode were copied
into the other's frontmatter; the `guest` field is each folder's own. Episodes
are keyed by video id, so the second folder of each pair overwrote the first's
row. On master's full ingest, 303 transcripts reported "ingested" became 272
episodes, with no error. 31 transcripts were gone, and 31 episodes showed one
guest's words under another's title and link. Because the two content hashes
differ, every later run re-embedded both, one after the other.

**My first fix was wrong, and the full-corpus run showed it.** I had found two
pairs in a 40-episode subset (`andy-raskin` / `andy-raskin_`, a copy, and
`benjamin-lauzier` / `benjamin-mann`), and fixed them by keeping the first
folder in sorted order and skipping the second. Ingesting all 303 then showed
31 pairs, and that the metadata's owner sorts first in only about half of
them. `alexander-embiricos` sorts before `nilan-peiris` but carries Nilan's
title, so "keep the first" kept Alexander's words under Nilan's title and
dropped Nilan's real episode. It lost as many episodes as master, just visibly.

**What the 31 pairs actually are**, compared over their first 3,000
characters:

- **7 are the same transcript twice** (similarity 0.98–1.00): `andy-raskin_`,
  `fei-fei`, `hamelshreya`, `ethan-evans-20`, `nicole-forsgren-20`,
  `wes-kao-20`, `yuhki-yamashata`. Skipping the copy loses nothing.
- **24 are different episodes** (similarity 0.11–0.33), one of them carrying
  the other's metadata.

**Fix.** Ingestion resolves shared ids before it starts
(`resolve_shared_video_ids`). Copies are skipped. For different episodes, the
metadata belongs to the folder whose guest the title names. That's
unambiguous in 18 pairs. In the other 6 the title can't tell them apart:
Elena Verna 2.0 / 3.0, Jake Knapp & John Zeratsky / 2.0, Melissa / Melissa
Tan, Shreyas Doshi / Live, Tomer Cohen / 2.0, Uri Levine / 2.0. Every folder
that doesn't own the metadata keeps its own guest and text, but loses the
borrowed id, URL, title and date. That's the same fallback the parser already
uses for episodes whose upstream metadata is empty: cited by guest, with no
link, and outside date filters. That's 30 transcripts, all now retrievable and
none mislabelled.

**A golden-set expectation that was fitted to the corrupted data.** The
retrieval eval asked "What's the ultimate guide to product-led sales?" and
expected guest `Elena Verna 3.0`. That's the title of Elena Verna's *2.0*
episode: `elena-verna-20` says "product-led sales" 50 times,
`elena-verna-30` twice. On master, 3.0's transcript had overwritten 2.0's row
under 2.0's title, so "Elena Verna 3.0" was the guest that row showed, and the
expectation was written to match. The fixed ingestion retrieves Elena Verna
2.0's episode first, and the expectation is corrected to match the transcript.

## Measured end to end, on the full corpus

The same 303 transcripts, in two databases on the same machine: one ingested
by master, one by all of the above (how it got there is below the table).
Then `python -m app.evals.run_eval` on each, with the corrected golden set.

| | master | fixed |
|---|---|---|
| Episodes | 272 | 296 |
| Episodes with no chunks | 26 | 0 |
| Chunks | 16,082 | 18,573 |
| Grounded answer rate, 14 in-domain questions | 100% | 100% |
| False-ground rate, 10 out-of-domain questions | 0% | 0% |
| Guest-match precision, 7 questions | 4/7 | 5/7 |

The guardrail that mattered most for a change that makes more text
retrievable is the false-ground rate, and it held at 0%: the closest
out-of-domain question stayed at 0.664 similarity against the 0.69 floor, and
the weakest in-domain one at 0.712. The one guest-match change is the Elena Verna question: master's database doesn't
contain her 2.0 episode at all.

master's run reported 303 ingested, 0 skipped, `ok`, for 272 episodes. The
fixed database was reached the way an existing deployment would be. First, an
ingest with the parser fix and the first, keep-the-first duplicate rule
(272 ingested, 31 skipped). Then one with the shared-id resolution over that
data: 41 re-ingested, the rest unchanged. Then
`python -m app.rag.ingest --prune`, which deleted exactly the 6 rows still
under the shared ids of the ambiguous pairs. The counts above are after that,
and a further run ingests 0 episodes and embeds nothing.

## What this shows

The checks earned their keep the first time they saw real data. Both defects
were invisible to every existing signal: the ingestion summary said `ok`, the
API served, and the 3-episode CI fixture is in the one layout the parser
already read. A check that asserts something about the *data* ("every episode
has chunks") caught what checks on the *code* couldn't.

It also shows the cost of fixing from a sample. The 40-episode subset had two
shared ids; the corpus had 31, and the fix that was right for two was wrong for
half of the rest. Only the full-corpus run, and a per-question look at the one
eval number that moved, found that -- and found an eval expectation that had
been written to match the bug.
