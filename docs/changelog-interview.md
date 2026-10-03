# Changelog, explained for interviews

One entry per change: what changed, why, and how to explain it.

## 1. Postman/Newman API collection

- **What changed:** `postman/` holds a 36-request collection covering all 13 API routes, with status, JSON-schema and response-time checks plus 404/422 negative cases, chained through a create → chat → read → delete run. A new CI job runs it with Newman against the real API and uploads a JUnit report.
- **Why:** the pytest suite tests the API in-process; this tests it from outside, over real HTTP, the way a client or a QA team would — and Postman/Newman is the tool most QA teams already use.
- **How it stays deterministic:** `backend/devtools/mock_llm.py` is a tiny server that speaks the OpenAI chat-completions protocol, so the app talks to it through its normal `openai_compat` provider. The agent loop, guards and database are real; only the model's choices are scripted. `devtools/seed_fixture.py` loads 3 short episodes with real embeddings instead of the hours-long full ingest.
- **How to explain the build:** the collection is generated from `postman/build_collection.py`, so tests are readable Python rather than JavaScript inside JSON strings; CI fails if the committed JSON drifts from the builder.
- **How I know the checks work:** with a 1 ms limit, exactly the 36 timing assertions fail; with two schemas changed to require a field the API doesn't return, exactly those 2 schema assertions fail.
