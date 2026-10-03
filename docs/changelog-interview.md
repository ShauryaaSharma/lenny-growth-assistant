# Changelog, explained for interviews

One entry per change: what changed, why, and how to explain it.

## 1. Postman/Newman API collection

- **What changed:** `postman/` holds a 36-request collection covering all 13 API routes, with status, JSON-schema and response-time checks plus 404/422 negative cases, chained through a create → chat → read → delete run. A new CI job runs it with Newman against the real API and uploads a JUnit report.
- **Why:** the pytest suite tests the API in-process; this tests it from outside, over real HTTP, the way a client or a QA team would — and Postman/Newman is the tool most QA teams already use.
- **How it stays deterministic:** `backend/devtools/mock_llm.py` is a tiny server that speaks the OpenAI chat-completions protocol, so the app talks to it through its normal `openai_compat` provider. The agent loop, guards and database are real; only the model's choices are scripted. `devtools/seed_fixture.py` loads 3 short episodes with real embeddings instead of the hours-long full ingest.
- **How to explain the build:** the collection is generated from `postman/build_collection.py`, so tests are readable Python rather than JavaScript inside JSON strings; CI fails if the committed JSON drifts from the builder.
- **How I know the checks work:** with a 1 ms limit, exactly the 36 timing assertions fail; with two schemas changed to require a field the API doesn't return, exactly those 2 schema assertions fail.

## 2. Test pyramid markers and coverage

- **What changed:** every test now carries one tier marker — `unit` (177), `api` (68), `integration` (51), `eval` (38) — plus `slow` (22) for tests that compute real embeddings. Markers are registered with `--strict-markers`; a collection hook fails the run if any test has no tier; `integration` is derived from `requires_db` so it can't drift. Coverage runs in CI with a floor of 72%.
- **Why:** so the suite can be run by layer (`pytest -m unit` in 15 s for fast feedback, `-m "not slow"` to skip the embedding model) and so coverage can't quietly fall.
- **A real bug the markers exposed:** a "unit" test of the blocked-artifact guard ran the real search tool, loaded the 130 MB embedding model (~7 s), then crashed on a `None` database — and still passed, because the tool swallows errors. It now uses the `empty_search` fake, and an autouse fixture makes any future unit test that loads the model fail loudly.
- **Why 72%:** that's what the full suite measured (72.33% of 2,259 statements), rounded down — not a target. The honest gaps are `rag/ingest.py` (0%), the LLM provider adapters (23–30%) and the live eval runners (43–58%).
- **How I know the floor works:** the unit tests alone reach 62.11% and the run fails with "Required test coverage of 72.0% not reached"; the full suite passes at 72.33%.
