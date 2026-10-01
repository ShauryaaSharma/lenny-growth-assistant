"""Load test for the backend API.

Models the traffic a busy UI generates: retrieval-heavy, with session
bookkeeping around it. The chat endpoint is tagged `llm` and excluded by
default (see locust.conf) because its latency is model inference -- 20-160 s
on a CPU-only 3B model, per docs/test-plan.md -- which would drown out every
number this test exists to measure. Run it separately with `--tags llm`.

    pip install -r loadtests/requirements.txt
    locust -f loadtests/locustfile.py --config loadtests/locust.conf --host http://localhost:8000

The run exits non-zero if the error rate or p95 latency breaches the limits
below, so it can be used as a gate rather than only read as a report.
"""

from __future__ import annotations

import logging
import os
import random

from locust import HttpUser, between, events, tag, task

# Questions phrased the way users actually ask, spanning grounded and
# out-of-domain cases so both retrieval outcomes are exercised.
QUERIES = [
    "What are early signs of product-market fit?",
    "How should a startup think about pricing?",
    "How do you build a growth loop?",
    "What makes onboarding effective?",
    "How do you hire your first product manager?",
    "What is a good retention benchmark for consumer apps?",
    "How do you run a good experiment?",
    "When should a founder stop doing sales?",
    "How do you write a strategy document?",
    "What is the best sourdough starter recipe?",
]

MAX_P95_MS = int(os.getenv("LOAD_MAX_P95_MS", "1500"))
MAX_FAILURE_RATIO = float(os.getenv("LOAD_MAX_FAILURE_RATIO", "0.01"))


class AssistantUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self) -> None:
        response = self.client.post("/api/sessions", json={"title": "load test"})
        self.session_id = response.json()["id"] if response.ok else None

    @task(6)
    def search(self) -> None:
        payload = {"query": random.choice(QUERIES), "top_k": 8}  # noqa: S311
        with self.client.post("/api/search", json=payload, catch_response=True) as response:
            if response.ok and "grounded" not in response.json():
                response.failure("search response is missing `grounded`")

    @task(2)
    def list_sessions(self) -> None:
        self.client.get("/api/sessions?limit=50")

    @task(2)
    def read_session(self) -> None:
        if self.session_id:
            self.client.get(f"/api/sessions/{self.session_id}", name="/api/sessions/[id]")

    @task(1)
    def config(self) -> None:
        self.client.get("/api/config")

    @task(1)
    def health(self) -> None:
        self.client.get("/health")

    @tag("llm")
    @task(1)
    def chat(self) -> None:
        if self.session_id:
            self.client.post(
                f"/api/sessions/{self.session_id}/chat",
                json={"message": random.choice(QUERIES)},  # noqa: S311
                name="/api/sessions/[id]/chat",
                timeout=300,
            )


@events.quitting.add_listener
def enforce_limits(environment, **_kwargs) -> None:
    total = environment.stats.total
    if total.num_requests == 0:
        logging.error("No requests were made -- is the host right?")
        environment.process_exit_code = 1
        return

    p95 = total.get_response_time_percentile(0.95)
    breaches = []
    if total.fail_ratio > MAX_FAILURE_RATIO:
        breaches.append(f"failure ratio {total.fail_ratio:.2%} > {MAX_FAILURE_RATIO:.2%}")
    if p95 > MAX_P95_MS:
        breaches.append(f"p95 {p95:.0f} ms > {MAX_P95_MS} ms")

    if breaches:
        logging.error("Load limits breached: %s", "; ".join(breaches))
        environment.process_exit_code = 1
    else:
        logging.info(
            "Load limits met: p95 %.0f ms, failure ratio %.2f%%", p95, total.fail_ratio * 100
        )
