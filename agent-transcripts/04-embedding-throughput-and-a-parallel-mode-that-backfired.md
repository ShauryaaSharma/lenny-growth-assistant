# 04 — Embedding throughput, and a "fix" that made it worse

**Context:** ingestion running against the real 303-episode corpus for the
first time, on the reference 16-thread CPU-only machine.

## The discovery

Roughly 2.5 minutes in, a progress check showed 4 episodes ingested, 163 chunks
embedded. Extrapolated, that is a ~4.5 hour run for the full corpus — a real
problem for the "first-run success rate" the PRD names as an operational
success metric, and a much worse first-boot experience than the README (at
that point) claimed.

## Diagnosis attempt 1 — thread count

`docker stats` showed the backend container at only ~137% CPU out of 16
available cores. A direct benchmark inside the container confirmed low
throughput (1.8–2.9 chunks/s across a few thread settings) but did not explain
*why* — the container clearly had access to all cores (`os.cpu_count() == 16`,
`os.sched_getaffinity` also returned 16), so this wasn't a Docker CPU-limit
misconfiguration.

## Diagnosis attempt 2 — fastembed's `parallel=0` multiprocessing mode

fastembed exposes a `parallel` argument where `0` is documented to fan work
across all available cores via a multiprocessing pool. This was tried as the
fix, on the reasoning that ONNX's own intra-op threading alone might not be
saturating the machine.

**Result: measurably worse, not better.** A clean benchmark with `parallel=0`
crashed outright:

```
RuntimeError: Worker PID: 1700 terminated unexpectedly with code -9
```

Exit code -9 is SIGKILL, and in a multiprocessing worker context on a
memory-constrained container, that is almost always the Linux OOM killer.
Diagnosis: `parallel=N` spins up N separate processes, each loading its own
full copy of the ONNX session and model weights. On this box's memory budget,
those workers were being killed mid-batch — which meant work was silently
being redone or serialized around the failures, explaining why throughput
under `parallel=0` was *not better* than the naive single-process version
measured earlier (0.73 vs ~1.1 chunks/s single-process, after accounting for
measurement noise from other things running at the same time).

**This was caught before it shipped as the default** by actually running the
isolated benchmark and reading its output, rather than assuming a
documented-as-fast library option is automatically the right choice
in a given resource envelope.

## The actual fix, and what it did and didn't solve

Reverted to single-process operation, with two real, verified improvements:

1. **`threads=os.cpu_count()`** passed explicitly to `TextEmbedding()`, giving
   ONNX's own intra-op parallelism the full core count within one process's
   memory budget, instead of the multiprocessing approach that broke under
   memory pressure.
2. **A persistent model cache directory** (`cache_dir` on a mounted Docker
   volume). This was a separate, unrelated defect found in the same
   investigation: the embedding model was being re-downloaded from Hugging
   Face on every single container restart (~40 seconds each time), because
   fastembed does not use the standard `HF_HOME` cache path. Fixed by pointing
   `cache_dir` at the `hfcache` volume already defined in `docker-compose.yml`.
   Verified directly: model load time dropped from ~41s to ~2s on the next
   restart.

**What this did not solve:** the underlying compute-bound throughput. Even
with correct single-process threading, the measured rate on this hardware
stayed at roughly 0.7–1 chunk/second — full-corpus ingestion is genuinely a
multi-hour operation on this specific CPU-only machine, not a
misconfiguration. Confirmed by watching `docker stats` show the container
legitimately pegged at 1500%+ CPU (15 of 16 cores) while still only producing
~0.74 chunks/second — this is the hardware's real transformer-inference speed,
not underutilization.

## The product decision that followed

Rather than keep chasing a config-level fix for a hardware-bound reality, this
was converted into an explicit, documented trade-off:

- The README's ingestion time estimate was corrected from an unverified
  "10–20 minutes" guess to the measured reality ("several hours" on this
  hardware profile), with an `INGEST_EPISODE_LIMIT` escape hatch documented
  for a fast smoke test.
- The PRD's risk table gained a new row naming this explicitly, including why
  it is not further fixable in software on this hardware, and what levers
  (GPU, smaller corpus) would actually move it.
- The system was already designed (from session 1) so that the knowledge base
  is marked `ready` as soon as the *first* chunk embeds, not when the full run
  finishes — so the multi-hour number is the time to full corpus coverage, not
  the time to a working demo.

## Why this is the most important entry in this log

This is the clearest example in the whole build of the difference between
"apply the fix that sounds right" and "verify the fix actually helped." The
multiprocessing change was a reasonable-sounding, documented library feature,
and it was actively harmful on this hardware. It was caught only because it
was benchmarked in isolation before being trusted, and the eventual answer was
not a clever software fix at all — it was an honest, documented product
trade-off about what CPU-only inference actually costs.
