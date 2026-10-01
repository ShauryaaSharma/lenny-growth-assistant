# Load tests

[Locust](https://locust.io) scenarios for the backend API. The traffic mix is
retrieval-heavy, the way a busy UI is: `POST /api/search` is half of all
requests, session reads and listing a third, `/api/config` and `/health` the
rest. Each
simulated user creates its own session first.

The chat endpoint is tagged `llm` and excluded by default. Its latency is
model inference (tens of seconds on a CPU-only 3B model), which would swamp
everything this test is for. Run it on its own with `--tags llm`.

## Running

Bring the stack up with some corpus ingested (a 40-episode subset is enough,
see the main README's fast smoke test), then:

```bash
cd backend
pip install -r loadtests/requirements.txt
locust -f loadtests/locustfile.py --config loadtests/locust.conf --host http://localhost:8000
```

`locust.conf` runs 20 users for 2 minutes, headless, and writes CSVs to
`loadtests/results/` (git-ignored). Override on the command line, e.g.
`--users 100 --spawn-rate 20 --run-time 1m`.

The run **exits non-zero** if the failure ratio exceeds 1% or p95 latency
exceeds 1500 ms. Both limits are env-tunable: `LOAD_MAX_FAILURE_RATIO`,
`LOAD_MAX_P95_MS`.

## Results

Measured 2026-10-02 on one 16-thread Windows laptop, with Locust, a single
uvicorn worker and Postgres (Docker, `pgvector/pgvector:pg16`) all on the same
machine. Corpus: 38 episodes, 2,215 chunks (2,111 embedded). Treat these as
relative numbers for this setup, not a capacity figure for production
hardware.

### What the first run found

At 100 users, endpoints that never touch the embedding model --
`/api/sessions/{id}`, `/api/config` -- had the same ~2 s p95 as search, and
even `/health` reached a 500 ms p95. That pattern means requests were queueing
behind something, not doing slow work themselves: `search()` ran the
synchronous, CPU-bound ONNX query embedding directly inside an async route,
blocking the event loop for every concurrent request. The fix moves it to a
worker thread (`asyncio.to_thread`, in `app/rag/retriever.py`).

### Before and after

100 users, 1 minute, 0 failures in both runs:

| | Before | After |
|---|---|---|
| Throughput | 53.6 req/s | 68.1 req/s |
| p50, all requests | 200 ms | 24 ms |
| p95, all requests | 2,000 ms | 1,300 ms |
| p50, `/api/search` | 260 ms | 35 ms |
| p95, `/health` | 500 ms | 88 ms |

20 users, 2 minutes (the default profile), 0 failures in both runs:

| | Before | After |
|---|---|---|
| p50 / p95, all requests | 16 / 63 ms | 15 / 50 ms |
| p50 / p95, `/api/search` | 19 / 74 ms | 18 / 60 ms |
| p99, all requests | 150 ms | 450 ms |

The p99 regression at 20 users is real and worth knowing about. Each ONNX call
is configured with one intra-op thread per core, so concurrent embeddings now
compete for the same cores instead of waiting in line. Capping intra-op
threads, or running more uvicorn workers, are the next things to measure.
The remaining 1.3 s p95 at 100 users is CPU saturation on a machine that was
also generating the load.
