# Architecture

## Contents

- [System overview](#system-overview)
- [Database schema](#database-schema)
- [Ingestion and retrieval flow](#ingestion-and-retrieval-flow)
- [Agent layer](#agent-layer)
- [API endpoints](#api-endpoints)
- [Security](#security)
- [Deployment topology](#deployment-topology)
- [Observability](#observability)

---

## System overview

Three services, one Compose file:

```
Next.js (3000) ──▶ FastAPI (8000) ──▶ Postgres + pgvector (5432)
                        │
                        ▼
                 Ollama (11434, host)  or  OpenAI-compatible cloud endpoint
```

The frontend never talks to Postgres, Ollama, or the cloud provider directly —
every external dependency is mediated by the FastAPI backend, which is the only
place credentials or connection strings exist.

---

## Database schema

```
episodes                    chunks                        sessions
─────────                   ──────                        ────────
id (pk)                     id (pk)                       id (pk)
video_id (unique)     ◀──── episode_id (fk, cascade)       title
guest                        ordinal                       user_metadata (jsonb)
title                        speaker                       created_at / updated_at
youtube_url                  start_seconds / end_seconds
publish_date                 text                          messages
duration_seconds              token_count                   ───────
description                  is_sponsor                     id (pk)
keywords (jsonb)              embedding (vector(384))        session_id (fk, cascade)
source_path                  tsv (generated tsvector)        role
content_hash                                                  content
ingested_at                                                   provider / model
                                                               citations (jsonb)
ingestion_runs                artifacts                       tool_calls (jsonb)
──────────────                ─────────                       latency_ms
id (pk)                       id (pk)                         created_at
status                        session_id (fk, cascade)
source                        message_id (fk, set null)
episodes_seen/ingested/skipped kind (markdown|html)
chunks_written                title
error                          content   ← post-sanitisation only
started_at / finished_at       sanitizer_report (jsonb)
                                created_at
```

**Design choices worth explaining:**

- **`chunks.tsv` is a generated column** (`GENERATED ALWAYS AS ... STORED`), not
  populated by the application. Postgres keeps it in sync automatically, so
  full-text search can never drift from the source text.
- **HNSW over IVFFlat for the vector index.** IVFFlat needs a training pass
  over existing data and performs poorly on an empty or small table — exactly
  the state the table is in for most of an ingestion run. HNSW has no training
  step and stays correct as rows are added incrementally.
- **`artifacts.content` stores only the sanitised form.** The raw model output
  is never persisted, so a future bug in the viewer cannot resurrect an unsafe
  payload from the database. See [Security](#security).
- **`content_hash` on episodes** makes ingestion idempotent: an unchanged
  episode is skipped on re-run, and re-ingestion after a corpus update only
  touches what actually changed.
- **Cascade deletes** on `session_id` and `episode_id` mean deleting a session
  or an episode cannot leave orphaned messages or chunks — there is no
  reconciliation job needed to keep the schema consistent.

Migration: [backend/alembic/versions/0001_initial_schema.py](../backend/alembic/versions/0001_initial_schema.py).

---

## Ingestion and retrieval flow

### Ingestion (`backend/app/rag/ingest.py`)

```
clone/pull corpus (git, shallow)
        │
        ▼
for each episodes/*/transcript.md:
        │
        ├─ parse YAML frontmatter + speaker turns  (rag/chunking.py)
        ├─ hash content → skip if unchanged from last run
        ├─ chunk on whole speaker turns, ~400 tokens, 80 overlap
        │      splitting only a monologue that alone exceeds the budget
        ├─ flag sponsor-read turns (kept, excluded from embedding)
        ├─ embed non-sponsor chunks (fastembed / bge-small-en-v1.5, ONNX, CPU)
        └─ upsert episode + chunks in one transaction per episode
        ▼
write an ingestion_runs row: counts, status, error, timing
```

**Why chunk on speaker turns rather than fixed windows:** a podcast answer is a
coherent unit of thought. A fixed 400-character window cuts mid-sentence,
producing chunks that embed fine but read as nonsense when quoted back to a
user as evidence. Turns are accumulated up to a token budget and carried over
with overlap, so a thought that straddles a boundary is retrievable from either
side.

**Why sponsor turns are flagged, not deleted:** roughly 5% of the corpus is ad
copy ("This episode is brought to you by..."). Left in the retrieval pool it
dominates results for commercial queries ("how should I price my product") and
produces confidently-cited nonsense. Flagging rather than deleting keeps the
data auditable — an operator can query `is_sponsor` directly rather than trust
that a deletion pass did the right thing.

**Why per-episode transactions:** a corpus-wide transaction means one malformed
file aborts everything ingested before it. A per-episode transaction means a
failure on episode 200 still leaves episodes 1–199 committed, and the failure
is recorded in `ingestion_runs` rather than silently dropped.

**Measured throughput and its consequence.** On the 16-thread CPU-only
reference machine, embedding the corpus proceeds at roughly 0.7–1 chunk/second
— the full 17,785-chunk corpus is a multi-hour run. This is CPU-bound ONNX
transformer inference; no batching or threading configuration found in testing
materially changed it (see the note in
[embeddings.py](../backend/app/rag/embeddings.py) about why `parallel=N`
multiprocessing was tried and rejected — it triggered OOM kills on this memory
profile and made throughput *worse*, not better). The system is designed around
this reality rather than against it: ingestion runs in the background from
boot, the knowledge base is marked `ready` as soon as the first chunk is
embedded, and `INGEST_EPISODE_LIMIT` lets an evaluator trade corpus breadth for
a fast smoke test.

### Retrieval (`backend/app/rag/retriever.py`)

```
query
  │
  ├──▶ vector arm:  cosine similarity over embedding, HNSW index, top 30
  ├──▶ text arm:    ts_rank over tsvector, GIN index, top 30
  │
  └──▶ Reciprocal Rank Fusion (k=60) over the two rank lists
              │
              ▼
      top-k fused results ──▶ grounding check
```

**Why hybrid, not vector-only:** embeddings blur rare tokens — a product name, a
metric name, a guest's surname. Lexical search catches those; vector search
catches paraphrase, which is most of how people actually phrase questions. RRF
combines two systems whose raw scores are not on comparable scales (cosine
similarity vs. `ts_rank`) without needing to calibrate one against the other.

**The grounding guard is deliberately NOT based on the fused RRF score.** RRF
scores are rank-derived: even a query the corpus cannot answer produces a
top-ranked hit with a plausible-looking fused score, because rank position
alone says nothing about absolute relevance. The guard instead checks raw
cosine similarity against `RETRIEVAL_MIN_SIMILARITY` (default 0.55), which *is*
an absolute signal. Below it, the search tool returns an explicit instruction
to decline rather than answer — see [Agent layer](#agent-layer).

---

## Agent layer

### Why not the Claude Agent SDK or Pi Coding Agent

The brief names these SDKs and separately requires the submitted demo to run on
local Ollama. Neither SDK executes against an Ollama endpoint. Building the
agent loop directly on one would have produced a system where the *demo path
the brief requires* was unimplementable — a worse outcome than deviating from
the suggested tooling.

Instead, `backend/app/llm/base.py` defines a minimal `LLMProvider` interface —
`chat()`, `health()` — and the agent loop (`backend/app/agent/runtime.py`) is
written against that interface only. `OllamaProvider` and
`OpenAICompatProvider` both implement it. The result: one code path, two
runtimes, and the cost is real — the SDKs' built-in session and tool plumbing
is hand-rolled here instead. That trade is made explicitly, not silently.

### The loop

```
system prompt + history + user message
        │
        ▼
   chat_with_fallback(messages, tools)
        │
        ├─ wants_tools? ──▶ execute each tool ──▶ append tool results ──▶ loop
        │
        └─ final answer:
              │
              ├─ substantive question, never searched? ──▶ force-search nudge, loop once
              ├─ searched but ungrounded? ──▶ ungrounded guard, loop once
              ├─ document request, no artifact created? ──▶ force-artifact nudge, loop once
              └─ otherwise: accept and return
```

Bounded at `MAX_ITERATIONS = 5`; if the model still hasn't converged, a final
no-tools turn asks it to answer in plain prose from whatever it has, so the
user never sees an empty response.

**Why three deterministic guards rather than trusting the system prompt:** on a
3B model, "the prompt tells it to always search first" (or "always render a
document as an artifact") is not reliable — it is probabilistic. Each guard
turns a should into a must. The forced-retrieval and ungrounded guards are the
mechanism behind the PRD's "0% of out-of-domain questions answered without
evidence" guardrail. The forced-artifact guard exists because of a defect
found live, not hypothetically: asked for "a one-page onboarding audit
checklist," `llama3.2:3b` searched correctly, then wrote the checklist as a
plain chat message instead of calling `create_artifact`. `wants_artifact()` in
`app/agent/prompts.py` matches a conservative keyword list against the user's
message, and the guard fires only when the answer was actually groundable —
never coercing an artifact out of an honest refusal. Full account, including a
second bug this fix surfaced (an ungrounded refusal was mislabeling itself as
`grounded: true` in the API response, by reusing that field as loop
bookkeeping) in
[agent-transcripts/07-artifact-routing-caught-by-manual-browser-testing.md](../agent-transcripts/07-artifact-routing-caught-by-manual-browser-testing.md).

**Ollama-specific reliability note.** Small quantised models sometimes emit a
well-formed tool call as plain text in the content field rather than populating
the structured `tool_calls` array. `OllamaProvider._salvage_tool_call` recovers
these by scanning content for a JSON object naming a *registered* tool — a
model hallucinating an unregistered name still falls through to a normal
answer. Measured directly against `llama3.2:3b` on the reference machine: the
model reliably returns structured tool calls for straightforward queries, so
salvage exists as a safety net for edge cases rather than the common path.

### Tools

Three, by design. Each additional tool measurably degrades routing accuracy on
small models, so the registry stays deliberately small (`backend/app/agent/tools.py`):

| Tool | Purpose |
|---|---|
| `search_transcripts` | The grounding primitive. Returns numbered excerpts or an explicit refusal instruction. |
| `write_ship30_essay` | Delegates to the essay pipeline; registers the result as an artifact rather than returning 1,250 words through the chat turn. |
| `create_artifact` | Sanitises and registers a markdown/HTML document. |

### The Ship 30 skill pipeline

```
topic
  │
  ├─ wide retrieval (top-14)
  ├─ outline: headline / hook / 3-4 sections / takeaway  (model call 1)
  ├─ draft: full essay following the outline               (model call 2)
  ├─ programmatic rubric check (word count, section count,
  │   citation count/validity, list presence, bold count)
  └─ if failed: targeted revision naming exactly what to fix (model call 3)
        keep the revision only if its rubric score is ≥ the original's
```

The principles the writer follows live in
[backend/app/skills/ship30/principles.md](../backend/app/skills/ship30/principles.md)
as data, not as a prompt string embedded in Python — editing house style is a
documentation change, and a reviewer can read exactly what the model was told
without reading code. The rubric check is what makes this a *skill* rather than
a prompt: a one-shot generation has no way to know it produced 400 words when
asked for 1,250. This does, and it fixes it.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness. No dependencies touched. |
| GET | `/health/deep` | Database, every configured LLM provider, and knowledge-base status, independently. |
| GET | `/api/config` | Provider, model, endpoint, and KB readiness — what the UI badge renders from. |
| POST | `/api/sessions` | Create a session. |
| GET | `/api/sessions` | List sessions with message counts, most recent first. |
| GET | `/api/sessions/{id}` | Full session detail: messages, citations, artifact summaries. |
| DELETE | `/api/sessions/{id}` | Delete a session (cascades to messages and artifacts). |
| POST | `/api/sessions/{id}/chat` | Run one agent turn. |
| GET | `/api/artifacts?session_id=` | List artifacts, optionally scoped to a session. |
| GET | `/api/artifacts/{id}` | Fetch one artifact's full sanitised content. |
| POST | `/api/search` | Retrieval only, no model in the loop — isolates "is this a retrieval problem or a model problem?" in one request. |

Every error response shares one envelope:

```json
{
  "error": {
    "code": "llm_unavailable",
    "message": "Cannot reach Ollama at http://host.docker.internal:11434.",
    "hint": "Is `ollama serve` running? Check GET /health/deep for provider status.",
    "request_id": "a1b2c3d4e5f6"
  }
}
```

`code` is machine-readable for the frontend, `message` and `hint` are for the
human at the keyboard, and `request_id` ties the error to the exact structured
log line that produced it.

---

## Security

Generated HTML is treated as untrusted, because it is: produced by a model
steered by user text, drawing on third-party corpus content. Prompt injection
in either could otherwise become stored XSS. Two independent layers, either
sufficient alone:

**Layer 1 — server-side sanitisation** (`backend/app/security/sanitize.py`).
Every artifact is parsed and rewritten against a strict tag/attribute allowlist
before it is ever persisted — `content` in the `artifacts` table is always
post-sanitisation. `<script>`, `<iframe>`, `<object>`, `<form>`, `<base>`,
`<link>`, `<meta>`, every `on*` handler, and `javascript:`/`data:text/html` URLs
are stripped. CSS is scrubbed of `@import` and remote `url()` references —
those are a real exfiltration channel (leaking the viewer's IP via a tracking
pixel in a stylesheet) that a naive allowlist misses. `data:` image URIs are
kept, since charts need to work offline and they reach no network.

**Layer 2 — sandboxed rendering** (`frontend/components/ArtifactViewer.tsx`).
HTML renders in an `<iframe sandbox="allow-scripts">` **without**
`allow-same-origin`. That combination puts the document on an opaque origin: it
cannot read the parent DOM, our cookies, or `localStorage`, and cannot make
same-origin requests back to the API. An injected `Content-Security-Policy` of
`default-src 'none'` blocks essentially all network egress from inside the
frame, so even a successful script injection has no channel to exfiltrate what
it sees.

Markdown never touches `dangerouslySetInnerHTML`. `react-markdown` parses to a
React element tree, and with no `rehype-raw` plugin installed, any HTML embedded
in markdown is rendered as literal text rather than executed.

Tested directly (`backend/tests/test_sanitize.py`, 25 cases): script tags,
event handlers, `javascript:`/`vbscript:`/`data:text/html` URLs, iframe/object/
embed/form/input/base injection, and CSS-based exfiltration, alongside
false-positive checks that ordinary CSS and markup pass through unmodified.

---

## Deployment topology

```
docker-compose.yml
├── postgres   (pgvector/pgvector:pg16, healthchecked, persistent volume)
├── backend    (python:3.12-slim; runs alembic migrations, then uvicorn;
│               seeds the KB in the background on first boot)
└── frontend   (Next.js standalone build, multi-stage image)

Ollama runs natively on the host, not in Compose — a 3B+ model competing with
Docker Desktop for RAM on a 16GB box is a worse trade than one extra
`ollama serve` process. The backend reaches it via `host.docker.internal`.
```

**Swapping Postgres for Supabase/Railway:** replace `DATABASE_URL` in `.env`.
No application code changes — SQLAlchemy speaks standard Postgres wire protocol
either way, and pgvector is available as an extension on both.

**Swapping the cloud model provider:** change `LLM_BASE_URL` / `LLM_MODEL` /
`LLM_API_KEY`. The adapter speaks the OpenAI `/v1/chat/completions` contract,
which is a de facto standard across Hugging Face's router, OpenAI, Groq, and
OpenRouter.

---

## Observability

Every log line is structured (`structlog`) and carries a `request_id` set by
middleware and threaded through a `contextvars` context — so one line from the
API, one from retrieval, one from the model call, and one from persistence for
the *same* chat turn all carry the same id, and `grep`ping it reconstructs the
whole turn. `LOG_FORMAT=json` switches to machine-readable output for shipping
to an aggregator; `console` (default) is for reading in a terminal.

`ingestion_runs` gives an audit trail independent of logs: status, counts, and
error for every ingestion attempt, queryable directly from Postgres.

`GET /health/deep` is the single request that answers "what, specifically, is
broken" — database, each configured LLM provider, and knowledge-base readiness
are reported independently, so a failure localises without reading logs at all.
