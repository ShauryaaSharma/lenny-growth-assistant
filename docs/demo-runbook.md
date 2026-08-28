# Demo runbook: start, test, record

Everything you need to (1) boot the app from a clean state, (2) exercise
every deliverable end-to-end, and (3) record the 2–3 minute submission video.
Follow it top to bottom the first time; after that, skip straight to
whichever section you need.

---

## Part 1 — Start the app

### 1.1 Prerequisites (one-time)

```bash
ollama pull llama3.2:3b
```

Confirm Ollama is actually serving it:

```bash
curl -s http://localhost:11434/api/tags
```

You should see `llama3.2:3b` in the JSON. If Ollama isn't running, start the
Ollama app/service first.

### 1.2 Configure

```bash
cp .env.example .env
```

The defaults are already correct for the local demo (`LLM_PROVIDER=ollama`,
`OLLAMA_MODEL=llama3.2:3b`). No edits needed unless you want the cloud
provider instead (see §1.6).

### 1.3 Boot everything

```bash
docker compose up --build
```

This builds and starts Postgres+pgvector, the FastAPI backend, and the
Next.js frontend. First boot also runs Alembic migrations and kicks off
corpus ingestion in the background.

### 1.4 Watch ingestion (optional but recommended before recording)

```bash
docker compose logs -f backend
```

Full-corpus ingestion (303 episodes) can take a while on a CPU-only box.
**For the video, don't wait for the full corpus** — a partial corpus already
answers real questions correctly, and the UI's own readiness banner is part
of what you're demoing. If you want a fast, deterministic subset instead:

```bash
docker compose exec -e INGEST_EPISODE_LIMIT=40 backend python -m app.rag.ingest
```

### 1.5 Verify it's actually ready

```bash
curl -s http://localhost:8000/health/deep | python -m json.tool
```

Confirm:
- `"status": "ok"`
- your provider (`ollama`) shows `"healthy": true`
- `knowledge_base.ready: true`

Then open **http://localhost:3000** in a browser.

### 1.6 (Optional) Switch to cloud provider

Only do this if you want to demonstrate the model toggle live. Edit `.env`:

```
LLM_PROVIDER=openai_compat
LLM_BASE_URL=https://router.huggingface.co/v1
LLM_MODEL=Qwen/Qwen3-32B
LLM_API_KEY=hf_your_token
```

Then:

```bash
docker compose restart backend
```

The provider badge in the UI header updates automatically. **Switch back to
`ollama` before recording** — the brief mandates the demo run on local Ollama.

---

## Part 2 — Test the app

Do this once before recording, so you know exactly what will happen on
camera and aren't discovering surprises live.

### 2.1 Automated tests

```bash
cd backend
pip install -r requirements-dev.txt
docker compose exec postgres createdb -U lenny lenny_test
python -m pytest -q
```

Expect **153 passed**. If some skip, check the printed reason — usually the
test database isn't reachable.

### 2.2 Evaluation harnesses (optional, good B-roll for the trade-off section)

```bash
python -m app.evals.run_eval
python -m app.evals.run_agent_eval
```

The first measures retrieval grounding against a 24-question golden set; the
second runs the real agent loop against 8 scripted conversations. Both print
a pass/fail summary — this is the "how do you know it works" evidence, not
just "it compiled."

### 2.3 Manual walkthrough (this is your video's shot list too)

Work through these in the browser at `http://localhost:3000`:

1. **New session** — click "+ New chat" in the sidebar. Confirm a fresh,
   empty conversation appears and the sidebar lists it.
2. **Grounded Q&A** — ask a real product/growth question, e.g.:
   > "What does Lenny's podcast say about finding product-market fit?"
   Confirm the answer cites an episode/guest and a timestamped link.
3. **Follow-up in the same session** — ask a pronoun-dependent follow-up
   (e.g. "How does that compare to onboarding?"). Confirm it uses session
   context correctly.
4. **Honest refusal** — ask something clearly outside the corpus, e.g.:
   > "What does Lenny's podcast say about the 2024 NBA finals?"
   Confirm it says the material doesn't cover it, instead of guessing.
5. **Ship 30 essay** — ask explicitly for an essay, e.g.:
   > "Write me a Ship 30 for 30 essay about product-market fit."
   Confirm: ~1,250 words, a hook, skimmable headings/bullets, a concrete
   takeaway, and claims traceable to the transcripts.
6. **Artifact viewer** — ask for a document, e.g.:
   > "Turn that into a one-page onboarding checklist as a document."
   Confirm it renders in the side panel (not inline as raw text), and that
   the panel is visually distinct from the chat.
7. **Model badge** — point out the provider/model shown in the UI header.
8. **Resilience (optional, good for the trade-off segment)** — stop Ollama
   (`docker compose exec backend curl ...` will now fail) and send a
   message; confirm the app returns a clear typed error instead of hanging
   or crashing. Restart Ollama afterward.
9. **Trace / observability (optional)** — after any chat turn, hit:
   ```bash
   curl -s http://localhost:8000/api/sessions/<session_id>/trace | python -m json.tool
   ```
   Shows recorded spans (LLM calls, tool calls, timings, token counts) —
   useful if you want one sentence on observability.

If every one of these behaves as described, you're ready to record.

---

## Part 3 — Demo video script (2–3 minutes, camera on)

The brief requires: camera on, explain the problem, show the product,
demonstrate local Ollama, and briefly cover one technical trade-off. Below
is a script timed to fit comfortably in 2:30–2:45. Practice it once
untimed, then once with a timer, before the take you keep.

**Setup before hitting record:** app already running (Part 1 done), browser
open to `http://localhost:3000` with an empty session ready, terminal
window ready but minimized, camera and mic checked.

---

**[0:00–0:25] The problem — camera on, talking head**

> "Hi, I'm [name]. This is my submission for the Forward Deployed Engineer
> take-home: the Lenny Growth Assistant.
>
> The scenario: a product and growth team wants their internal assistant to
> answer questions from Lenny's Podcast — hundreds of episodes of operator
> interviews — without anyone having to know what a prompt or a vector
> database is. They want grounded answers, reusable written content, and
> documents that just show up rendered, not raw code they have to copy
> somewhere else."

**[0:25–1:00] Show the product — screen share, narrate while acting**

> "Here's the app. I'll start a new chat and ask a real question."

- Type: *"What does Lenny's podcast say about finding product-market fit?"*
- While it answers: "Notice this cites the actual episode and guest it came
  from — every answer is grounded in the transcript corpus, not the model's
  own training data."
- Ask a quick follow-up to show session memory.

**[1:00–1:30] Ship 30 essay + artifact viewer**

> "It also has a dedicated content skill. If I ask for a Ship 30 for
> 30-style essay..."

- Type: *"Write a Ship 30 for 30 essay about product-market fit."*
- While it generates: "This isn't just a longer prompt — the writing
  principles from the Ship 30 guide are encoded in a proper skill with a
  quality check, not a one-off instruction."
- Then: "And if I ask for something to hand off as a document..."
- Type: *"Turn that into a one-page onboarding checklist."*
- Point at the artifact panel rendering beside the chat: "This renders in a
  sandboxed viewer — generated HTML is treated as untrusted, so it can't
  run scripts or exfiltrate anything, similar to how Claude's own Artifacts
  work."

**[1:30–1:55] Local Ollama — mandatory demonstration**

> "This entire demo is running against a local model — no API key, nothing
> leaving this machine. It's a 3-billion-parameter Llama model served by
> Ollama on my own CPU."

- Point at the provider badge in the header showing `ollama` / `llama3.2:3b`.
- Optionally flip to the terminal for one second: `docker compose ps` /
  `ollama list` — proof it's actually local, not a cloud call in disguise.
- "The same code path also talks to any OpenAI-compatible cloud endpoint —
  switching is one environment variable, no code changes."

**[1:55–2:30] One technical trade-off — camera on or split screen**

> "One trade-off worth calling out: the brief suggests building the agent
> layer on the Claude Agent SDK or Pi Coding Agent, but it also requires the
> demo to run on local Ollama — and neither of those SDKs can actually drive
> an Ollama endpoint. Building directly on one of them would have made the
> mandatory local demo impossible.
>
> So I built a small provider-agnostic agent loop instead, behind one
> `LLMProvider` interface, so the exact same tool-calling code path serves
> both local and cloud models. The cost is that I had to hand-build the
> session and tool-calling plumbing those SDKs would normally give me for
> free — but the benefit is one code path, fully swappable, with no
> vendor lock to a provider that couldn't even run the required demo."

**[2:30–2:45] Close**

> "Everything here — the PRD, architecture doc, design doc, tests, and the
> agent transcripts showing real bugs I found and fixed along the way — is
> in the GitHub repo linked in my submission. Thanks for watching."

---

### Recording tips

- Do a full dry run first with the timer visible — it's easy to run past
  3 minutes once you're narrating live typing.
- If Ollama is slow mid-recording (CPU contention with anything else
  running), have a **pre-typed, already-answered** session ready as a
  fallback you can scroll to instead of waiting on camera.
- Upload to YouTube (unlisted is fine unless the form says otherwise) and
  paste the link into the submission form.
