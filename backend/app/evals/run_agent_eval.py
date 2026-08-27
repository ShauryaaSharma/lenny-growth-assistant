"""Agent evaluation harness -- the counterpart to run_eval.py, one layer up.

Run against the real, running system (real model, real tool registry, real
guards):

    docker compose exec backend python -m app.evals.run_agent_eval

Skip the expensive essay scenario for a fast iteration loop:

    docker compose exec backend python -m app.evals.run_agent_eval --exclude-slow

What this measures that neither the unit tests nor the retrieval eval can:

    - Tool-call correctness: given a real message, does the real model (via
      the real LLMProvider and the real guards) actually call the tool it
      should. `tests/test_agent_routing.py` verifies the guard *logic* against
      a scripted FakeProvider; this verifies the *model* actually produces the
      tool call the logic is there to enforce.
    - Refusal correctness: does an out-of-domain question actually get refused
      in the final answer, not just marked ungrounded internally.
    - Artifact correctness: is a requested document actually rendered as an
      artifact, and is an ungrounded document request actually still refused.
    - Redundancy / efficiency: did any tool get called more times than a
      single correct turn should need. This is the direct, automated
      regression guard for agent-transcripts/09 -- the ctx.searched bug that
      silently doubled the cost of every successful essay or artifact. That
      defect produced a *correct final answer*, so no output-only check would
      have caught it; only counting tool calls does.

This is slower than run_eval.py by design -- it calls the real, required local
model, which is the whole point. Expect single-turn scenarios to take
10-160s each depending on CPU contention, and the one `slow=True` essay
scenario to take several minutes. Run this before merging any change to the
agent loop, the system prompt, or the guards -- not on every commit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field

from app.agent.runtime import run_agent
from app.config import get_settings
from app.db.session import get_sessionmaker
from app.evals.agent_scenarios import AGENT_SCENARIOS, REFUSAL_MARKERS, AgentScenario, ScenarioTurn
from app.llm.base import ChatMessage, LLMError
from app.logging import configure_logging


@dataclass
class TurnResult:
    message: str
    checks: dict[str, bool]
    correct: bool
    tool_calls: list[str]
    grounded: bool
    has_artifact: bool
    content_snippet: str
    latency_ms: int


@dataclass
class ScenarioResult:
    name: str
    category: str
    slow: bool
    turns: list[TurnResult]
    correct: bool  # all turns correct
    notes: str


def _looks_like_refusal(content: str) -> bool:
    low = content.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def score_turn(turn: ScenarioTurn, result) -> TurnResult:
    """Pure scoring over an AgentResult -- no I/O, fully unit-testable.

    Correctness checks are evaluated against *executed* tool calls only
    (`ok=True`), not every attempt. A call the runtime blocked (the
    ungrounded-artifact guard, the redundant-artifact guard) is recorded in
    `ctx.tool_log` under the same tool name so the attempt stays visible in
    the trace, but it had no real effect -- counting it toward "was the
    forbidden tool called" or "was this tool called too many times" would
    score a safety net doing its job as a defect. `tool_names` below is kept
    unfiltered for the human-readable trace; `executed` drives every check.
    """
    tool_names = [t["tool"] for t in result.tool_calls]
    executed = [t["tool"] for t in result.tool_calls if t.get("ok", True)]
    checks: dict[str, bool] = {}

    if turn.expect_tools_any_of:
        checks["expected_tool_called"] = any(t in executed for t in turn.expect_tools_any_of)

    if turn.forbidden_tools:
        checks["no_forbidden_tools"] = not any(t in executed for t in turn.forbidden_tools)

    for tool, max_count in turn.max_tool_calls.items():
        count = executed.count(tool)
        checks[f"no_redundant_{tool}"] = count <= max_count

    if turn.expect_grounded is not None:
        checks["grounded_as_expected"] = result.grounded == turn.expect_grounded

    if turn.expect_artifact is not None:
        has_artifact = len(result.artifacts) > 0
        checks["artifact_as_expected"] = has_artifact == turn.expect_artifact

    if turn.expect_refusal_language:
        checks["refusal_language_present"] = _looks_like_refusal(result.content)

    return TurnResult(
        message=turn.message,
        checks=checks,
        correct=all(checks.values()) if checks else True,
        tool_calls=tool_names,
        grounded=result.grounded,
        has_artifact=len(result.artifacts) > 0,
        content_snippet=result.content[:160],
        latency_ms=result.latency_ms,
    )


async def run_scenario(scenario: AgentScenario) -> ScenarioResult:
    """Run one scenario. A provider failure on one scenario (a real risk on a
    CPU-only local model under load -- Ollama's 180s timeout is not generous)
    must not crash the whole suite and lose every other scenario's result;
    it is recorded as a failed turn instead, same as a wrong routing decision.
    """
    sessionmaker = get_sessionmaker()
    session_id = uuid.uuid4()
    history: list[ChatMessage] = []
    turn_results: list[TurnResult] = []

    async with sessionmaker() as db:
        for turn in scenario.turns:
            try:
                result = await run_agent(db, session_id, turn.message, history)
            except LLMError as exc:
                turn_results.append(
                    TurnResult(
                        message=turn.message,
                        checks={"provider_call_succeeded": False},
                        correct=False,
                        tool_calls=[],
                        grounded=False,
                        has_artifact=False,
                        content_snippet=f"<provider error: {type(exc).__name__}: {exc}>",
                        latency_ms=0,
                    )
                )
                break  # later turns in this scenario depend on this one's reply
            turn_results.append(score_turn(turn, result))
            history.append(ChatMessage(role="user", content=turn.message))
            history.append(ChatMessage(role="assistant", content=result.content))

    return ScenarioResult(
        name=scenario.name,
        category=scenario.category,
        slow=scenario.slow,
        turns=turn_results,
        correct=all(t.correct for t in turn_results),
        notes=scenario.notes,
    )


def summarize(results: list[ScenarioResult]) -> dict:
    passed = all(r.correct for r in results)
    failures = [
        f"{r.name}: turn {i+1} failed checks {[k for k, v in t.checks.items() if not v]}"
        for r in results
        for i, t in enumerate(r.turns)
        if not t.correct
    ]
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r.correct)

    return {
        "passed": passed and bool(results),
        "failures": failures,
        "n_scenarios": len(results),
        "category_pass_rate": {
            cat: sum(v) / len(v) for cat, v in by_category.items()
        },
        "results": [
            {**asdict(r), "turns": [asdict(t) for t in r.turns]} for r in results
        ],
    }


def print_report(summary: dict) -> None:
    print()
    print("=" * 72)
    print("AGENT SCENARIO EVAL REPORT")
    print("=" * 72)
    for r in summary["results"]:
        mark = "PASS" if r["correct"] else "FAIL"
        slow_tag = " [slow]" if r["slow"] else ""
        print(f"[{mark}] {r['category']:16} {r['name']}{slow_tag}")
        if r["notes"]:
            print(f"       note: {r['notes']}")
        for i, t in enumerate(r["turns"], start=1):
            tmark = "ok" if t["correct"] else "FAIL"
            print(
                f"       turn {i} [{tmark}] tools={t['tool_calls']} grounded={t['grounded']} "
                f"artifact={t['has_artifact']} {t['latency_ms']}ms"
            )
            print(f"                \"{t['message']}\"")
            if not t["correct"]:
                failed_checks = [k for k, v in t["checks"].items() if not v]
                print(f"                failed: {failed_checks}")
                print(f"                reply: {t['content_snippet']!r}")
    print("-" * 72)
    print(f"Scenarios run: {summary['n_scenarios']}")
    for cat, rate in summary["category_pass_rate"].items():
        print(f"  {cat:18} {rate:.0%}")
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
    parser.add_argument("--json", metavar="PATH", help="Write the full summary as JSON to this path.")
    parser.add_argument(
        "--exclude-slow", action="store_true",
        help="Skip scenarios that invoke the Ship 30 essay pipeline (several minutes each).",
    )
    parser.add_argument("--list", action="store_true", help="List scenarios and exit without running.")
    args = parser.parse_args()

    scenarios = [s for s in AGENT_SCENARIOS if not (args.exclude_slow and s.slow)]

    if args.list:
        for s in scenarios:
            tag = " [slow]" if s.slow else ""
            print(f"[{s.category:16}] {s.name}{tag}")
        return

    settings = get_settings()
    configure_logging(settings.log_level, "console")

    started = time.perf_counter()
    results = asyncio.run(_run_all(scenarios))
    elapsed = time.perf_counter() - started

    summary = summarize(results)
    print_report(summary)
    print(f"Total wall time: {elapsed:.1f}s\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.json}")

    sys.exit(0 if summary["passed"] else 1)


async def _run_all(scenarios: list[AgentScenario]) -> list[ScenarioResult]:
    return [await run_scenario(s) for s in scenarios]


if __name__ == "__main__":
    main()
