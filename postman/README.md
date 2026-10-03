# API collection

`lenny.postman_collection.json` exercises every route in `backend/app/api/`
in one ordered run: it creates a session, chats in it (grounded answer,
refusal, document, streamed turn), reads back what was saved, then deletes
it. Ids pass between requests through collection variables.

Every request checks its status code, validates the response body against
a JSON schema, and checks response time against the environment's limits
(`maxResponseMs`, and `maxChatMs` for full agent turns). Negative cases cover
404s, 422 validation failures and malformed ids, each checked against the
shared error envelope and its `code`.

## Run it locally

From `backend/`, against a fresh database:

```bash
alembic upgrade head
python -m devtools.seed_fixture           # 3 episodes, real embeddings
python -m devtools.mock_llm --port 9999   # in another terminal
```

Start the API pointed at the mock (in a third terminal):

```bash
INGEST_ON_STARTUP=false LLM_PROVIDER=openai_compat LLM_BASE_URL=http://127.0.0.1:9999/v1 LLM_API_KEY=mock uvicorn app.main:app --port 8000
```

Then, from the repository root:

```bash
npx newman run postman/lenny.postman_collection.json -e postman/local.postman_environment.json
```

The mock LLM (`backend/devtools/mock_llm.py`) speaks the OpenAI
chat-completions protocol, so the app reaches it through its ordinary
`openai_compat` provider: the agent loop, tools, guards and persistence are
all real; only the model's choices are scripted, which is what makes the
run deterministic.

## Editing the collection

The JSON is generated. Edit `build_collection.py` and regenerate:

```bash
python postman/build_collection.py
```

CI runs it with `--check` and fails if the committed JSON is out of date.
The generated file still imports into the Postman app as a normal
collection.
