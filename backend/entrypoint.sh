#!/usr/bin/env bash
# Fail fast and loudly — a half-migrated database is worse than no startup.
set -euo pipefail

echo "[entrypoint] applying database migrations..."
alembic upgrade head

echo "[entrypoint] starting API on :8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-config app/logging_uvicorn.json
