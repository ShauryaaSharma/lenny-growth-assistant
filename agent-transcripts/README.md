# Agent transcripts

This project was built with Claude Code as the coding agent, working
interactively with the author over roughly two sessions. This folder documents
that process per the brief's requirement to include coding-agent logs, failed
attempts, and how they were corrected.

The raw tool-call transcript is not reproduced verbatim here — it is long and
mostly mechanical (file writes, docker commands, log tails). Instead this is an
honest engineering log: what was tried, what failed, why, and what changed.
Nothing below is retrospective tidying; each entry reflects a real failure
encountered while building this system on this hardware.

## Session 1 — scaffolding and de-risking

**Goal:** stand up the repo skeleton and de-risk the two things most likely to
break the mandatory local-Ollama demo: model tool-calling and transcript
chunking against the real corpus, before building the rest of the system on
top of untested assumptions.

**What happened, in order:**

1. Read the take-home brief from the supplied `.docx`, researched the transcript
   corpus (found `ChatPRD/lennys-podcast-transcripts`, 303 episodes with rich
   YAML frontmatter) and the Ship 30 for 30 writing method, and proposed an
   architecture — see [00-planning-and-corpus-selection.md](00-planning-and-corpus-selection.md).
2. Identified upfront that the brief's suggested agent SDKs (Claude Agent SDK,
   Pi Coding Agent) cannot execute against a local Ollama endpoint, which the
   brief separately requires for the demo. Decided to build against a custom
   `LLMProvider` interface instead, and to document the deviation rather than
   hide it.
3. Installed Ollama, cloned the transcript corpus, and immediately ran the
   chunker against all 303 real files rather than a synthetic sample — see
   [01-chunking-defects-found-by-testing-early.md](01-chunking-defects-found-by-testing-early.md).
   **This surfaced three real defects before any code was built on top of the
   chunker**, each of which would have silently degraded answer quality rather
   than raising an error:
   - An 800-token chunk target exceeded the `bge-small-en-v1.5` embedding
     model's 512-token window, which would have silently truncated (and made
     unretrievable) the tail of every long chunk.
   - Four episodes ship with empty `video_id`/`youtube_url` frontmatter and
     were being dropped by a naive required-field check, discarding real
     transcript content.
   - Ollama binds to IPv4 only; `http://localhost:11434` fails while
     `http://127.0.0.1:11434` succeeds — a footgun worth documenting in the
     README's troubleshooting section rather than letting an evaluator hit it
     cold.
4. Built the backend layer by layer (config, schema, retrieval, LLM providers,
   agent loop, skills, API) and the frontend (chat, artifact viewer, sandbox).
5. Verified the HTML sanitiser against a deliberate attack list (script tags,
   event handlers, `javascript:`/`data:text/html` URLs, CSS exfiltration via
   `@import`/remote `url()`) before trusting it, rather than assuming an
   allowlist-based sanitiser is safe by construction.

## Session 2 — build, ingest, and test against reality

**Goal:** get the full system actually running end-to-end against Docker,
Postgres, and the real 303-episode corpus, and write the tests that verify
behaviour rather than just structure.

**What happened, in order:**

1. `docker compose build backend` initially reported success in one background
   run but had actually failed with an `apt-get` network timeout — see
   [02-concurrent-downloads-starved-each-other.md](02-concurrent-downloads-starved-each-other.md).
   Root cause: running the Ollama model pull and the Docker build concurrently
   over a shared ~2 MB/s link starved both. Fix: serialize large downloads.
2. Two FastAPI startup crashes on first real boot, both caused by the same
   underlying interaction between `from __future__ import annotations` and a
   204 No Content route — see
   [03-fastapi-204-response-model-crash.md](03-fastapi-204-response-model-crash.md).
3. Discovered mid-ingestion that embedding throughput was going to take
   multiple hours for the full corpus on this CPU-only machine, then made it
   measurably *worse* by trying to fix it with multiprocessing — see
   [04-embedding-throughput-and-a-parallel-mode-that-backfired.md](04-embedding-throughput-and-a-parallel-mode-that-backfired.md).
   This is the most consequential finding in the whole build: the "obvious"
   performance fix was actively harmful on a memory-constrained box, and the
   real fix (single-process ONNX threading + a persistent model cache) only
   fixed the *repeated cold-start* cost, not the fundamental CPU-bound
   inference speed. That reality is now documented honestly in the README and
   PRD risk table instead of glossed over with an optimistic estimate.
4. Docker Desktop and the in-progress ingestion were killed by an unplanned
   machine restart mid-session — see
   [05-machine-restart-recovery.md](05-machine-restart-recovery.md). Verified
   that the idempotent, content-hashed ingestion design (built in session 1,
   before this failure ever happened) meant recovery was "restart the
   containers and re-run the ingest command," not "start over."
5. Wrote the DB-backed test suites (session isolation, retrieval ranking)
   against a real Postgres instance rather than mocks, which caught a real
   SQLAlchemy async identity-map staleness bug in the *test* itself — see
   [06-sqlalchemy-identity-map-staleness-in-tests.md](06-sqlalchemy-identity-map-staleness-in-tests.md).

## What this log is trying to demonstrate

Per the brief: "we are evaluating your judgment and ability to direct, verify,
and improve AI-assisted work — not whether every line was typed manually."
Concretely, that judgment showed up as:

- Testing the chunker against the real 303-episode corpus on day zero instead
  of a synthetic sample, which is what actually surfaced the token-window and
  missing-metadata defects.
- Treating "the build succeeded" as a claim to verify, not a fact to trust —
  which is what caught the silent `apt-get` failure.
- Benchmarking a performance fix before trusting it, which is what caught the
  multiprocessing regression before it shipped as the default.
- Making the ingestion design idempotent *before* anything went wrong, which
  is what made an unplanned machine restart a non-event instead of a lost day.
