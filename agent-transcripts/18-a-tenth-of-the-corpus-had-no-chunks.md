# 18 — A tenth of the corpus had no chunks

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

## Defect 2 — one episode silently overwrote another

**Symptom.** 40 transcripts, 38 episodes, and no duplicate video ids in the
table.

**Root cause.** Two pairs of folders upstream carry the same video id.
`andy-raskin_` is a copy of `andy-raskin` (the files differ by 6 bytes).
`benjamin-mann` holds Benjamin Mann's own transcript, about AI, but its title,
URL and video id are copied from `benjamin-lauzier`'s marketplace episode.
Episodes are keyed by video id, so the second of each pair took over the
first's row: Benjamin Lauzier's episode was replaced by Benjamin Mann's
transcript, under Lauzier's title and link. And because the two content hashes
differ, every later ingestion run re-embedded both, one after the other.

**Fix.** The first folder in sorted order is kept; the other is logged
(`duplicate_video_id`, with both paths) and counted as skipped. A second run
re-embeds nothing. For Andy Raskin that loses nothing. For Benjamin Mann it
means his episode is not in the knowledge base until the upstream metadata is
corrected, which is better than serving his words under another guest's title
and link. The validation check now counts *distinct* video ids, and a separate
warning lists the shared ones, so it stays visible until then.

## What this shows

The checks earned their keep the first time they saw real data. Both defects
were invisible to every existing signal: the ingestion summary said `ok`, the
API served, and the 3-episode CI fixture is in the one layout the parser
already read. A check that asserts something about the *data* ("every episode
has chunks") caught what checks on the *code* couldn't.
