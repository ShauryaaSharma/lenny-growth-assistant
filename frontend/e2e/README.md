# End-to-end tests

Playwright tests for the critical user journeys, in a real browser against
the real API and database:

| Journey | What it checks |
|---|---|
| Start a session | A new chat is selected in the sidebar, empty, with the composer ready |
| Ask a question | The answer cites `[1]`; the source list shows that source, guest and a working timestamped link; the question names the chat |
| Guest filter | A question naming a guest gets an answer and sources from only that guest |
| Date filter | "...since 2024" gets an answer and sources from only 2024 episodes |
| Document | The document renders in the viewer, its source tab shows the markdown, and the chip on the reply reopens it |
| Refusal | An off-topic question is answered "not covered", with no sources |

The model is replaced by `backend/devtools/mock_llm.py`, which the app reaches
through its normal OpenAI-compatible provider. Everything else is real, which
is what makes each answer exact and the run deterministic.

The guest and date journeys check that the filter reaches the answer and its
sources. That a filter changes *ranking* — including against a corpus where
other guests outrank the one asked about — is covered by the integration tests
in `backend/tests/test_search_filters.py`.

## Run locally

Start the backend exactly as for the API collection (see `postman/README.md`):
seeded fixture corpus, mock LLM on port 9999, API on port 8000. Then:

```bash
cd frontend
npm ci
npx playwright install chromium
npx playwright test
```

Playwright starts `npm run dev` itself, or reuses one already running on
port 3000. On failure it keeps a trace, a screenshot and a video under
`test-results/`; open a trace with `npx playwright show-trace <path>`.
