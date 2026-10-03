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

## 3. Contract testing from the OpenAPI schema

- **What changed:** Schemathesis runs in CI against the live API, generating requests for every non-LLM endpoint from `/openapi.json` and checking each response: no 5xx, only documented status codes, bodies matching their schemas (errors included), invalid input rejected. Final run: 11 operations, 803 generated requests, all 7 checks pass.
- **What it found (agent-transcripts 14–17):** a session titled with one NUL character was a 500 (Postgres can't store U+0000; now a 422 on every stored string); a 500's `request_id` was `-`, in the response and the log (a middleware reset it before the error handler ran); `?offset=2^63` overflowed Postgres's bigint `OFFSET` (now bounded at the bigint max); and the schema documented FastAPI's default 422 shape instead of the API's real envelope, plus no 404s at all.
- **Why the fourth matters:** a client generated from `/openapi.json` would have parsed every error wrongly. Schemathesis didn't flag it because FastAPI's default doesn't require its `detail` field — so a passing contract check is only as strict as the schema it checks.
- **How I know the checks work:** with a required field added that the API never sends, every 422 failed the schema check; with the fixes reverted, 17 of the 18 new regression tests fail (the 18th is a boundary test meant to pass both ways).
- **One tool choice to explain:** Schemathesis is pinned to 3.x because 4.x needs a native extension that this Windows machine's Application Control policy blocks; FastAPI emits OpenAPI 3.1, which 3.x supports behind `--experimental=openapi-3.1`.

## 4. Playwright end-to-end tests

- **What changed:** `frontend/e2e/` has 6 Playwright journeys — start a session, ask and see the citation, filter by guest, filter by date, generate a document that renders in the viewer, and an off-topic refusal — run headless in CI against the production build, with traces, screenshots and video uploaded only for failures.
- **How it's deterministic:** the same mock LLM as the API collection, reached through the app's normal provider, so the UI, API and database are real and each answer is exact. No retries are configured, so a flaky test fails rather than hiding.
- **A design finding:** the UI has no filter controls — filters come from what the user types (guest detection, or the model passing `since`/`until`). So the filter journeys type the question, as a user would; the mock passes date bounds the way a capable model would.
- **A bug caught in my own mock:** I first mapped "before 2024" to `until=2024`, but a bare year means the whole year, so it should be `until=2023`; fixed and covered by its unit tests.
- **How I know the tests work:** with the UI's source list deliberately disabled, exactly the 3 journeys that read sources failed (each with a screenshot); the refusal test, which asserts there are none, still passed.

## 5. SQL checks on the knowledge base

- **What changed:** `backend/app/validation/` runs 12 SQL checks on the ingested data: episode count against the corpus, no episode without chunks, no gaps in chunk numbering, no chunk without its episode, no missing or wrong-size embeddings, plausible publish dates, both search indexes valid, no chunk with zero keyword-searchable words, and a sample of chunks found again by their own embedding and their own words. `python -m app.validation` prints a pass/warn/fail table and exits 1 on any failure; CI runs it right after seeding the fixture corpus.
- **Why:** these are the failures that make retrieval quietly worse without raising an error. A chunk with no embedding is invisible to vector search; an empty full-text vector is invisible to keyword search; a half-finished index build leaves an index that exists but is never used. Nothing crashes; answers just get worse.
- **Why check what the schema already enforces:** the foreign key and the `vector(384)` column type can be bypassed (a bulk load with triggers off) or drift (someone changes `EMBEDDING_DIM` without a migration). One query each is cheaper than finding out from bad answers.
- **Fail vs warn:** an episode with no publish date is a warning, not a failure. It's still answerable, but date filters can never match it.
- **How I know the checks work:** 11 tests, each breaking the data one way (orphans via `session_replication_role = replica`, a dropped index, a stopword-only chunk, a 768 config against a 384 column) and asserting that exactly that check fails. Against the seeded fixture, the CLI reports 12/12 passing; with `--expected-episodes 4` it fails that one check and exits 1.
