"""Turn a Locust CSV into a Markdown summary labelled with how it was run.

    python loadtests/summarize.py loadtests/results --users 20 --spawn-rate 5 \
        --run-time 2m --episodes 40

Reads `run_stats.csv` (written by `--csv loadtests/results/run`) and the
corpus counts the run script saved, and prints the date, machine, load
profile and per-endpoint latencies. A number is only quotable with all of
those next to it, so they are written into the same file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = ("Request Count", "Failure Count", "Requests/s", "50%", "95%", "99%")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def memory_gb() -> str:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return f"{int(line.split()[1]) / 1024 / 1024:.0f} GB"
    except OSError:
        pass
    return "unknown"


def hardware() -> str:
    return (f"{cpu_model()}, {os.cpu_count()} logical CPUs, {memory_gb()} RAM, "
            f"{platform.system()} {platform.release()}")


def fmt(column: str, value: str) -> str:
    if column == "Requests/s":
        return f"{float(value):.1f}"
    return value if column.endswith("Count") else f"{value} ms"


def summarize(results: Path, users: int, spawn_rate: int, run_time: str,
              episodes: int, now: datetime | None = None,
              machine: str | None = None) -> str:
    rows = list(csv.DictReader((results / "run_stats.csv").open(encoding="utf-8")))
    corpus_file = results / "corpus.json"
    corpus = json.loads(corpus_file.read_text()) if corpus_file.exists() else {}
    when = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "## Load test",
        "",
        f"- **When:** {when}",
        f"- **Where:** {machine or hardware()}; Locust, the API (one uvicorn worker) "
        "and Postgres on the same machine, via docker compose",
        f"- **Load:** {users} users, spawn rate {spawn_rate}/s, {run_time}; chat excluded "
        "(`--exclude-tags llm`)",
        f"- **Corpus:** {episodes}-episode subset: {corpus.get('episodes', '?')} episodes, "
        f"{corpus.get('chunks', '?')} chunks ({corpus.get('embedded_chunks', '?')} embedded)",
        "",
        "| Endpoint | Requests | Failures | Req/s | p50 | p95 | p99 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        name = ("**All requests**" if row["Name"] == "Aggregated"
                else f"`{row['Type']} {row['Name']}`")
        cells = " | ".join(fmt(c, row[c]) for c in COLUMNS)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--spawn-rate", type=int, required=True)
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    print(summarize(args.results, args.users, args.spawn_rate, args.run_time, args.episodes))


if __name__ == "__main__":
    main()
