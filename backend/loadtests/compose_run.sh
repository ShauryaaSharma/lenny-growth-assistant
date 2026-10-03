#!/usr/bin/env bash
# One load-test run against the docker compose stack, start to finish:
#
#   bring up Postgres + the API -> ingest a corpus subset -> validate it ->
#   run Locust headless -> write CSV, HTML and a labelled Markdown summary.
#
# Run from the repository root:
#
#   bash backend/loadtests/compose_run.sh
#
# Tunable through the environment: USERS (20), SPAWN_RATE (5), RUN_TIME (2m),
# EPISODES (40). Results land in backend/loadtests/results/. Exits with
# Locust's code, so a breached p95 or failure-rate limit fails the run.
# Leaves the stack running; `docker compose down -v` removes it.
set -euo pipefail

USERS="${USERS:-20}"
SPAWN_RATE="${SPAWN_RATE:-5}"
RUN_TIME="${RUN_TIME:-2m}"
EPISODES="${EPISODES:-40}"
API="http://localhost:8000"
RESULTS=backend/loadtests/results

if [[ ! -f .env ]]; then
  cp .env.example .env
fi
# The override sets the ingest limit and log level for this run only;
# everything else comes from .env, as for a normal `docker compose up`.
export EPISODES
compose() { docker compose -f docker-compose.yml -f backend/loadtests/docker-compose.loadtest.yml "$@"; }

echo "== starting postgres and the API"
compose up -d --build --wait postgres backend

echo "== waiting for ingestion of $EPISODES episodes"
status=""
for _ in $(seq 1 180); do
  status=$(curl -sf "$API/health/deep" \
    | python -c "import json,sys; print(json.load(sys.stdin)['knowledge_base'].get('last_run_status') or '')" \
    || true)
  [[ "$status" == "ok" || "$status" == "failed" ]] && break
  sleep 10
done
if [[ "$status" != "ok" ]]; then
  echo "ingestion did not finish (last status: '${status:-none}')" >&2
  compose logs backend | tail -50 >&2
  exit 1
fi

mkdir -p "$RESULTS"
curl -sf "$API/health/deep" \
  | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)['knowledge_base']))" \
  > "$RESULTS/corpus.json"
cat "$RESULTS/corpus.json"

echo "== validating the knowledge base before measuring it"
# No --expected-episodes: inside the container the check counts the
# transcripts ingestion would pick up under the same INGEST_EPISODE_LIMIT.
compose exec -T backend python -m app.validation

echo "== load: $USERS users, spawn rate $SPAWN_RATE/s, $RUN_TIME"
code=0
(cd backend && locust -f loadtests/locustfile.py --config loadtests/locust.conf --host "$API" \
   --users "$USERS" --spawn-rate "$SPAWN_RATE" --run-time "$RUN_TIME" \
   --csv loadtests/results/run --html loadtests/results/report.html) || code=$?

(cd backend && python loadtests/summarize.py loadtests/results --users "$USERS" \
   --spawn-rate "$SPAWN_RATE" --run-time "$RUN_TIME" --episodes "$EPISODES") \
  | tee "$RESULTS/summary.md"
exit "$code"
