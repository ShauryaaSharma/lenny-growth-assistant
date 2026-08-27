# Manual test plan

Automated tests (`backend/tests/`) cover chunking, sanitisation, and agent
routing logic. This plan covers what they cannot: the actual UI, the actual
Ollama integration, and the actual browser sandbox behaviour. Run through this
before any release-quality claim about the system.

Prerequisite: `docker compose up --build`, then wait for `knowledge_base.ready`
in `GET /health/deep` (or for the amber banner in the UI to clear).

---

## 1. First boot and health

| # | Step | Expected |
|---|---|---|
| 1.1 | Fresh clone, `cp .env.example .env`, `docker compose up --build` | All three containers start; no manual steps beyond these two commands |
| 1.2 | `curl http://localhost:8000/health` | `{"status":"ok"}` immediately, even while ingestion is still running |
| 1.3 | `curl http://localhost:8000/health/deep` | `database.healthy: true`; `providers[]` includes `ollama`; `knowledge_base` present |
| 1.4 | Open `http://localhost:3000` before ingestion finishes | Amber "Building the knowledge base..." banner visible; composer still usable |
| 1.5 | Wait for the first episode to embed, reload | Banner disappears without a manual refresh being required (polling) |
| 1.6 | Send a chat message *while ingestion is still running in the background* | Answered correctly, but expect materially higher latency than the README's steady-state estimate — ingestion and chat inference compete for the same CPU cores. Verified directly during this build: a query that normally answers in ~20–30s took ~160s under concurrent ingestion load. This is expected, not a bug; retry after ingestion finishes for a representative timing. |

## 2. Sessions

| # | Step | Expected |
|---|---|---|
| 2.1 | Load the app with zero prior sessions | One session auto-created, composer focused |
| 2.2 | Click **+ New chat** twice | Two sessions in the sidebar, most recent selected |
| 2.3 | Send a message in session A, switch to session B, send a different message | Each session shows only its own messages on reload |
| 2.4 | Delete the active session | Falls back to the next session in the list, or creates a fresh one if none remain |
| 2.5 | Refresh the browser mid-conversation | Full history reloads from `GET /api/sessions/{id}`, including citations |

## 3. Grounded Q&A

| # | Step | Expected |
|---|---|---|
| 3.1 | Ask *"What are signs of product-market fit?"* | Answer cites at least one guest/episode; citations expand to show timestamp and a working YouTube link |
| 3.2 | Ask a follow-up using a pronoun, e.g. *"What did he say about metrics?"* | Answered using conversation context, not treated as a fresh unrelated query |
| 3.3 | Ask something the corpus does not cover, e.g. *"What's the best sourdough starter recipe?"* | Assistant states plainly that Lenny's Podcast transcripts don't cover this — no fabricated answer, no citations |
| 3.4 | Ask a trivial message: *"hi"* | Answered directly, with **no** `search_transcripts` call (check `tool_calls` in the response or the backend logs) |
| 3.5 | Click a citation link | Opens the correct YouTube video at the cited timestamp in a new tab |

## 4. Ship 30 essay

| # | Step | Expected |
|---|---|---|
| 4.1 | Ask *"Write a Ship 30 essay about retention as a growth lever"* | An artifact chip appears; chat reply is a short description, not the essay text itself |
| 4.2 | Open the artifact | ~1,050–1,450 words, a `#` headline, 3–5 `##` sections, at least one list, inline `[n]` citations |
| 4.3 | Check citations against the article | Every `[n]` in the text corresponds to a real, cited source in the artifact metadata |
| 4.4 | Ask for an essay on an out-of-corpus topic | Assistant declines to write it and explains the corpus doesn't support it, rather than producing an ungrounded essay |

## 5. Artifacts and the security sandbox

| # | Step | Expected |
|---|---|---|
| 5.1 | Ask for *"a one-page onboarding audit checklist as a document"* | Markdown artifact renders with headings, checklist items |
| 5.2 | Ask for *"the same thing as a styled HTML page"* | HTML artifact renders inside the sandboxed panel with its own styling |
| 5.3 | Open browser DevTools while an HTML artifact is open, inspect the iframe | `sandbox="allow-scripts"` present, **no** `allow-same-origin` |
| 5.4 | With an HTML artifact open, check the Network tab | No requests originate from the iframe's document beyond what the parent page itself made |
| 5.5 | Switch the artifact panel to the **Source** tab | Raw sanitised HTML is shown as plain text, not executed |
| 5.6 | (Adversarial) Ask the assistant to *"include a script tag that shows an alert"* in an HTML artifact | No alert fires; if anything was stripped, the "N items removed" disclosure appears and lists it |

## 6. Provider toggle and resilience

| # | Step | Expected |
|---|---|---|
| 6.1 | Note the badge in the sidebar header | Shows `Local · llama3.2:3b` (or the configured model) |
| 6.2 | Stop Ollama (`taskkill` / kill the process), send a message | Chat returns a typed error: code `llm_unavailable`, with a hint to start Ollama; UI shows the red banner with that hint, not a blank failure |
| 6.3 | Restart Ollama, send another message | Recovers without restarting the backend |
| 6.4 | Set `LLM_PROVIDER=openai_compat` with a valid `LLM_API_KEY`, restart backend | Badge switches to `Cloud · <model>`; a message is answered by the cloud provider (`message.provider` in the response confirms it) |
| 6.5 | Set `LLM_PROVIDER=openai_compat` with an **empty** key | `GET /health/deep` shows that provider unhealthy with `detail: "LLM_API_KEY not set"`; a chat attempt returns a typed `llm_auth` error, not a crash |
| 6.6 | Set `LLM_FALLBACK_PROVIDER=ollama` while the cloud key is invalid | Chat request falls back to Ollama and answers; response `provider` field shows `ollama`, proving the fallback actually fired rather than just being configured |

## 7. Database resilience

| # | Step | Expected |
|---|---|---|
| 7.1 | `docker compose stop postgres`, then `GET /health/deep` | `database.healthy: false`, overall `status: "error"`; backend process itself stays up |
| 7.2 | With Postgres stopped, send a chat message | Typed 5xx error, not an unhandled exception or hung request |
| 7.3 | `docker compose start postgres` | Backend recovers on the next request (connection pool reconnects; no restart needed) |

## 8. Accessibility spot-check

| # | Step | Expected |
|---|---|---|
| 8.1 | Tab from page load with no mouse | Skip link appears first, jumps focus straight to the composer |
| 8.2 | Tab through the sidebar, chat, and an open artifact panel | Visible focus ring at every stop, order matches visual layout |
| 8.3 | Trigger an error banner with a screen reader running | Announced immediately (`role="alert"`) |
| 8.4 | Resize to a mobile width (< 768px) | Sidebar becomes a drawer with a scrim; an open artifact becomes a full-screen overlay |
