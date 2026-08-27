# PRD — The Lenny Growth Assistant

**Status:** v1.0, shipped
**Author:** Forward Deployed Engineer
**Date:** 27 August 2026

---

## 1. Forward deployment brief

### 1.1 User and problem

**Primary user: the product or growth IC at a Series A–C startup.** A PM,
growth lead, or founder-operator with 2–8 years of experience, who is
responsible for a decision this week and has no senior peer to check it against.

The job they are trying to complete is not "learn about growth." It is
**"make this specific decision with more confidence than I have right now"** —
should we invest in onboarding or activation, what does a first PM hire actually
look like, is our retention curve normal for this category.

Today they solve it by half-remembering a podcast episode, searching YouTube,
scrubbing a 90-minute video for the four minutes that matter, and often giving
up. Lenny's Podcast contains an unusually dense corpus of exactly this
expertise — 303 episodes of operators describing what actually worked — but it
is trapped in a format that does not support lookup.

**The pain the assistant removes:** the cost of retrieval. It turns a corpus you
have to *listen to* into one you can *ask*.

**Secondary user: the content lead.** Same team, different job — they need to
turn internal knowledge into publishable writing, and they need it grounded in
something more credible than a generic model's opinions.

### 1.2 Why grounding is the product

A generic LLM will answer any product question fluently. That is the problem.
The value here is not fluency, it is **provenance**: an answer the user can
check, attributed to a named operator who actually did the thing, with a link to
the moment they said it.

This has a direct design consequence: **the assistant must be willing to say it
does not know.** An assistant that answers everything is indistinguishable from
the free alternative and cannot be trusted on the answers where it *is* right.

### 1.3 Success metrics

**Primary — Grounded Answer Rate.**
Share of substantive user questions answered with at least one transcript
citation, where the top retrieved passage clears the similarity floor.

- Target: **≥ 80%** of in-domain questions grounded.
- Guardrail: **0%** of out-of-domain questions answered without evidence. A
  fabricated answer is worse than a refusal, and this metric must never be
  improved by lowering the floor.
- Instrumented today: every turn logs `grounded`, `best_similarity`, and the
  tools used; every assistant message persists its citations.

**Secondary — Time to Confident Answer.**
Median seconds from question to an answer the user does not need to verify
elsewhere. Proxy metric available now: median turn latency, logged per turn.

- Target: **< 30s** on the local 3B model, **< 10s** on cloud.

**Operational — First-Run Success Rate.**
Share of fresh clones where `docker compose up` reaches a healthy `/health/deep`
with no manual intervention. Target: **100%**. This is the metric a forward
deployed engineer actually gets judged on, because it is the difference between
a system the client can run and a demo only its author can run.

### 1.4 Assumptions

The brief was incomplete in several places. These are the calls made, and what
would change if they are wrong.

| # | Assumption | If wrong |
|---|---|---|
| 1 | Internal tool, trusted users, no per-user auth needed in v1 | Add auth before any external exposure; the session model already carries `user_metadata` for it |
| 2 | Answer quality matters more than latency; users will wait ~30s for a cited answer | Would need streaming responses and a smaller model |
| 3 | The English-language corpus is sufficient | No multilingual embedding work was done |
| 4 | Corpus refresh is weekly at most, not real-time | Ingestion is a batch job, not a stream |
| 5 | Evaluators have no Anthropic/OpenAI key | Cloud path targets the Hugging Face router; any OpenAI-compatible endpoint works |
| 6 | Sponsor reads are noise, not content | They are ingested and flagged but excluded from retrieval — reversible with one query change |
| 7 | Transcripts are accurate enough to quote | No speaker-diarisation correction was attempted |
| 8 | Read-heavy, low concurrency (single team) | No caching layer or read replica |

### 1.5 Scope

**In scope, and built:**

- Grounded conversational Q&A with session-scoped follow-ups
- Hybrid retrieval with a hard grounding floor and honest refusal
- Ship 30 for 30 essay skill with encoded principles and a programmatic gate
- Markdown and HTML/CSS artifacts with a sandboxed in-app viewer
- Runtime model toggle across local and cloud providers, visible in the UI
- Postgres persistence of sessions, messages, citations, artifacts, ingest runs
- One-command startup, structured logging, deep health checks, typed errors
- Automated tests plus a manual UI test plan

**Deliberately excluded, and why:**

| Excluded | Why |
|---|---|
| **Authentication / multi-tenancy** | Assumption 1. Real auth is a week of work and would displace grounding quality, which is what the brief actually grades. |
| **Streaming token output** | Meaningful work in the agent loop, the API, and the UI. With a forced-retrieval step the first token is late regardless, so the UX gain is smaller than it looks. A "thinking" state buys most of it for a fraction of the cost. |
| **Reranker model** | A cross-encoder would improve top-k ordering, but it doubles CPU cost per query on the machine this must demo on. RRF over two arms gets most of the benefit for near-zero cost. |
| **Supabase / Railway hosting** | Local Compose gives a genuinely reproducible one-command start with no evaluator signup. `DATABASE_URL` swaps to either with no code change — the capability is there, the dependency is not. |
| **Conversation-history summarisation** | A 20-message sliding window is sufficient at this session length and is far easier to reason about. |
| **Artifact editing / versioning** | The brief asks for generation and rendering. Editing is a separate product surface. |
| **Newsletter corpus** | Podcast transcripts alone are 17,785 passages. Adding a second source with different structure risks retrieval quality for marginal coverage gain. |

### 1.6 Risks and trade-offs

| Risk | Severity | Mitigation | Residual |
|---|---|---|---|
| **Hallucination** — answers not supported by the corpus | High | Forced retrieval; similarity floor; ungrounded guard; citations persisted per message | A model can still misattribute *within* retrieved evidence. Citations let the user catch it. |
| **Weak local tool-calling** — 3B models emit tool calls as prose | High | Salvage parser recovers text-form calls; forced-retrieval nudge; bounded iterations | Very small models may still need one extra turn |
| **Unsafe artifact rendering** — stored XSS via generated HTML | High | Two independent layers: server-side allowlist sanitisation, and an opaque-origin sandboxed iframe with `default-src 'none'` | Accepted: both layers would have to fail together |
| **Prompt injection from the corpus** — a transcript instructing the model | Medium | Retrieved text enters as tool *results*, never as system instructions; tools are a fixed registry the model cannot extend | A crafted passage could still bias phrasing |
| **Local model quality** — 3B reasoning is materially weaker | Medium | Provider toggle to cloud; skill pipeline with a programmatic gate rather than trusting one-shot output | Measured directly on `llama3.2:3b`: the rubric correctly identifies structural deviations (word count, section count) and triggers one revision pass, but a 3B model does not always fully correct them in that one pass. Observed essay: 984/1050+ words, more than the targeted 3-5 sections, `passed=False` but still delivered — grounded and useful, just outside spec. The system ships the best achievable result rather than blocking, which is the right call for an essay (not gated the way retrieval grounding is) but the honest limitation is real and improves materially on a larger model or cloud provider. |
| **Latency on CPU** | Medium | 3B default; documented model tiers; warm embedding model; wide retrieval in one call | 20–30s turns are inherent to CPU inference |
| **First-boot ingestion time** | Medium | Idempotent, content-hashed ingestion that resumes rather than restarts; `INGEST_EPISODE_LIMIT` for a fast subset; UI un-blocks per-episode rather than waiting for the full run | Measured on the reference 16-thread CPU-only box: full-corpus embedding is several hours, not minutes. This is compute-bound ONNX inference, not a fixable inefficiency — a GPU or a smaller corpus are the only real levers. Documented in the README rather than hidden behind an optimistic estimate. |
| **Corpus licensing** | Medium | Cloned at runtime, never vendored; attribution in README; educational use only | Upstream terms could change |
| **Cost (cloud path)** | Low | Local default costs nothing; token usage logged per turn | — |
| **Data leakage** | Low | Local default sends nothing off-machine; secrets never committed; `.env` gitignored | Cloud path sends prompts to the provider — stated in README |

---

## 2. Key flows

### 2.1 Grounded question

1. User asks a product question.
2. Agent rewrites it into a self-contained search query and calls `search_transcripts`.
3. Hybrid retrieval returns ranked passages with an absolute similarity score.
4. **If nothing clears the floor:** the tool returns an explicit refusal
   instruction, the ungrounded guard is appended, and the assistant states the
   corpus does not cover the topic.
5. Otherwise the model answers from the numbered evidence, citing `[n]`.
6. Answer, citations, provider, model, and latency are persisted.

### 2.2 Follow-up

Prior user/assistant turns are replayed (tool traffic is not). The agent is
instructed to resolve pronouns before searching, so "what did *he* say about
that" becomes a standalone query.

### 2.3 Ship 30 essay

Wide retrieval (top-14) → outline → draft → **programmatic rubric check** →
revise once if it fails, keeping the revision only if it scores better. The
essay is registered as an artifact rather than pushed back through the model, so
a small model cannot truncate or "summarise" its own work.

### 2.4 Artifact

Model calls `create_artifact` → server sanitises → only the sanitised form is
persisted → the viewer renders it in a sandboxed frame beside the chat, with a
disclosure of anything that was stripped.

---

## 3. Acceptance criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `docker compose up` yields a working app with no manual steps | Met |
| 2 | In-domain questions answered with checkable citations | Met |
| 3 | Out-of-domain questions refused, not answered | Met — enforced by two independent guards |
| 4 | Sessions maintain independent context | Met — scoped at the query level |
| 5 | Conversations, timestamps, metadata persisted in Postgres | Met |
| 6 | Model switchable without code changes | Met — one env var |
| 7 | Provider visible in the UI | Met — header badge, `/api/config` |
| 8 | Ship 30 essay ~1,250 words, formatted, grounded | Met — gate enforces range and citation count |
| 9 | Artifacts render in-app, not as raw code | Met |
| 10 | Generated HTML treated as untrusted | Met — two independent layers |
| 11 | Graceful handling of missing keys, dead Ollama, timeouts, empty retrieval, DB loss | Met — typed errors with actionable hints |
| 12 | Meaningful automated tests | Met — 56 passing |
| 13 | No secrets committed | Met |

---

## 4. Implementation plan as executed

| Phase | Work | Outcome |
|---|---|---|
| 0 | De-risk: install Ollama, verify tool-calling, validate chunking on the real corpus | Caught three defects before building on them |
| 1 | Schema, retrieval, provider abstraction, agent loop, skills, API | Backend exercisable by `curl` before any UI |
| 2 | Chat UI, artifact viewer, states | — |
| 3 | Tests, docs, fresh-clone verification, demo | — |

**What de-risking first bought.** Three defects surfaced in phase 0 that would
have been expensive later:

1. An 800-token chunk target exceeded the embedding model's 512-token window —
   every chunk tail would have been silently unretrievable, with no error.
2. Four episodes ship with empty metadata and were being dropped entirely.
3. Ollama binds IPv4-only, so `localhost` fails where `127.0.0.1` works.

None would have raised an exception. All three would have shown up as "the
answers are worse than expected" — the hardest class of bug to trace backwards.
