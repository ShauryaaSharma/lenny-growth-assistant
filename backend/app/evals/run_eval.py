"""Retrieval evaluation harness.

Run against a real, running knowledge base:

    docker compose exec backend python -m app.evals.run_eval

Or on the host, with DATABASE_URL pointed at the running Postgres:

    cd backend && python -m app.evals.run_eval

What this measures that the unit test suite cannot:

    - Grounded Answer Rate on in-domain questions (the PRD's primary success
      metric), against the REAL retriever and REAL embedded corpus -- not a
      scripted FakeProvider.
    - False-Ground Rate on out-of-domain questions -- the PRD's guardrail
      ("0% of out-of-domain questions answered without evidence"). This is
      arguably the more important number: a system that grounds everything
      including nonsense is not trustworthy on the answers it gets right.
    - Guest-match precision: for in-domain questions naming an expected guest,
      does that guest's episode actually surface in the top results.
    - Retrieval latency (p50/p95), since that's the one part of a chat turn
      this harness can measure without needing a live, slow local model.

This intentionally does NOT run the model or the agent loop -- only
`rag.retriever.search()`. That keeps a full run to a few seconds even on a
CPU-only machine, so there is no excuse not to run it after any change to
chunking, embeddings, or the retrieval SQL. Agent-level behaviour (does the
model actually refuse when told to?) is covered separately by the scripted
tests in `tests/test_agent_routing.py` and by the manual test plan.

Exit code is non-zero if either guardrail is violated, so this is safe to
wire into CI once a CI runner with a seeded database exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass

from app.config import get_settings
from app.db.session import get_sessionmaker
from app.evals.golden_set import GOLDEN_SET, GoldenQuestion
from app.logging import configure_logging
from app.rag.retriever import search

# Below this in-domain grounded rate, or above this out-of-domain false-ground
# rate, the run fails CI. Matches the PRD's stated success metric and guardrail.
MIN_GROUNDED_RATE = 0.80
MAX_FALSE_GROUND_RATE = 0.0


@dataclass
class QuestionResult:
    query: str
    category: str
    expected_grounded: bool
    actual_grounded: bool
    best_similarity: float
    latency_ms: int
    top_guests: list[str]
    expect_guest: str | None
    guest_matched: bool | None  # None when not applicable
    correct: bool


async def evaluate_question(db, q: GoldenQuestion) -> QuestionResult:
    started = time.perf_counter()
    result = await search(db, q.query)
    latency_ms = int((time.perf_counter() - started) * 1000)

    top_guests = [c.guest for c in result.chunks[:5]]
    guest_matched: bool | None = None
    if q.expect_guest is not None:
        guest_matched = result.grounded and any(
            q.expect_guest.lower() == g.lower() for g in top_guests
        )

    return QuestionResult(
        query=q.query,
        category=q.category,
        expected_grounded=q.expect_grounded,
        actual_grounded=result.grounded,
        best_similarity=round(result.best_similarity, 4),
        latency_ms=latency_ms,
        top_guests=top_guests,
        expect_guest=q.expect_guest,
        guest_matched=guest_matched,
        correct=(result.grounded == q.expect_grounded),
    )


async def run(golden_set: list[GoldenQuestion] = GOLDEN_SET) -> dict:
    sessionmaker = get_sessionmaker()
    results: list[QuestionResult] = []
    async with sessionmaker() as db:
        for q in golden_set:
            results.append(await evaluate_question(db, q))
    return summarize(results)


def summarize(results: list[QuestionResult]) -> dict:
    in_domain = [r for r in results if r.category == "in_domain"]
    out_of_domain = [r for r in results if r.category == "out_of_domain"]

    grounded_rate = (
        sum(r.actual_grounded for r in in_domain) / len(in_domain) if in_domain else None
    )
    false_ground_rate = (
        sum(r.actual_grounded for r in out_of_domain) / len(out_of_domain)
        if out_of_domain
        else None
    )
    guest_checks = [r for r in in_domain if r.expect_guest is not None]
    guest_precision = (
        sum(bool(r.guest_matched) for r in guest_checks) / len(guest_checks)
        if guest_checks
        else None
    )
    latencies = [r.latency_ms for r in results]

    passed = True
    failures: list[str] = []
    if grounded_rate is not None and grounded_rate < MIN_GROUNDED_RATE:
        passed = False
        failures.append(
            f"Grounded Answer Rate {grounded_rate:.0%} is below the "
            f"{MIN_GROUNDED_RATE:.0%} floor from the PRD's success metric."
        )
    if false_ground_rate is not None and false_ground_rate > MAX_FALSE_GROUND_RATE:
        passed = False
        failures.append(
            f"False-Ground Rate {false_ground_rate:.0%} on out-of-domain questions "
            f"violates the PRD's guardrail (must be {MAX_FALSE_GROUND_RATE:.0%})."
        )

    return {
        "passed": passed,
        "failures": failures,
        "n_questions": len(results),
        "grounded_answer_rate": grounded_rate,
        "false_ground_rate": false_ground_rate,
        "guest_match_precision": guest_precision,
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": (
                sorted(latencies)[int(0.95 * len(latencies))] if latencies else None
            ),
            "max": max(latencies) if latencies else None,
        },
        "results": [asdict(r) for r in results],
    }


def print_report(summary: dict) -> None:
    print()
    print("=" * 72)
    print("RETRIEVAL EVAL REPORT")
    print("=" * 72)

    for r in summary["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        guest_note = ""
        if r["expect_guest"] is not None:
            guest_note = f"  guest_match={r['guest_matched']}"
        print(
            f"[{mark}] {r['category']:14} grounded={r['actual_grounded']!s:5} "
            f"sim={r['best_similarity']:.3f}  {r['latency_ms']:4}ms{guest_note}"
        )
        print(f"       {r['query']}")
        if not r["correct"]:
            print(
                f"       ^ expected grounded={r['expected_grounded']}, "
                f"got {r['actual_grounded']}"
            )

    print("-" * 72)
    gar = summary["grounded_answer_rate"]
    fgr = summary["false_ground_rate"]
    gmp = summary["guest_match_precision"]
    lat = summary["latency_ms"]
    print(f"Questions run:            {summary['n_questions']}")
    print(f"Grounded Answer Rate:      {gar:.0%}" if gar is not None else "Grounded Answer Rate:      n/a")
    print(f"False-Ground Rate:         {fgr:.0%}" if fgr is not None else "False-Ground Rate:         n/a")
    print(f"Guest-match precision:     {gmp:.0%}" if gmp is not None else "Guest-match precision:     n/a")
    print(f"Retrieval latency (ms):    p50={lat['p50']:.0f}  p95={lat['p95']:.0f}  max={lat['max']}")
    print("-" * 72)
    if summary["passed"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        for f in summary["failures"]:
            print(f"  - {f}")
    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", metavar="PATH", help="Also write the full summary as JSON to this path."
    )
    parser.add_argument(
        "--list", action="store_true", help="List the golden set and exit, without running it."
    )
    args = parser.parse_args()

    if args.list:
        for q in GOLDEN_SET:
            guest = f"  (expect: {q.expect_guest})" if q.expect_guest else ""
            print(f"[{q.category:14}] {q.query}{guest}")
        return

    settings = get_settings()
    configure_logging(settings.log_level, "console")

    summary = asyncio.run(run())
    print_report(summary)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.json}")

    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
