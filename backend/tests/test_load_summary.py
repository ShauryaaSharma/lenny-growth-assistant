"""The load-test summary: every number it prints is labelled with how it was
measured, and it reads Locust's CSV columns correctly."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from loadtests.summarize import summarize

pytestmark = pytest.mark.unit

HEADER = ("Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
          "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
          "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%")


def test_summary_labels_the_run_and_reads_the_percentiles(tmp_path):
    (tmp_path / "run_stats.csv").write_text("\n".join([
        HEADER,
        "POST,/api/search,600,0,35,40,10,900,2000,5.0,0,35,40,45,50,60,74,90,120,300,900,900",
        ",Aggregated,1200,2,20,25,1,900,900,10.04,0.0,20,22,25,30,40,63,80,150,300,900,900",
    ]))
    (tmp_path / "corpus.json").write_text(json.dumps(
        {"episodes": 38, "chunks": 2215, "embedded_chunks": 2111}))

    text = summarize(tmp_path, users=20, spawn_rate=5, run_time="2m", episodes=40,
                     now=datetime(2026, 10, 3, 9, 30, tzinfo=UTC), machine="Test CPU, 4 cores")

    assert "**When:** 2026-10-03 09:30 UTC" in text
    assert "**Where:** Test CPU, 4 cores" in text
    assert "20 users, spawn rate 5/s, 2m" in text
    assert "38 episodes, 2215 chunks (2111 embedded)" in text
    assert "| `POST /api/search` | 600 | 0 | 5.0 | 35 ms | 74 ms | 120 ms |" in text
    assert "| **All requests** | 1200 | 2 | 10.0 | 20 ms | 63 ms | 150 ms |" in text
