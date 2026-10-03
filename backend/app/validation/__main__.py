"""`python -m app.validation`: run the knowledge-base checks; exit 1 on failure."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date

from app.db.session import dispose_engine, get_sessionmaker
from app.validation.checks import DEFAULT_MIN_PUBLISH_DATE, Check, run_all

MARK = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"{MARK[c.status]}  {c.name:<{width}}  {c.detail}")
        lines.extend(f"      {'':<{width}}  e.g. {e}" for e in c.examples)
    failed = sum(not c.ok for c in checks)
    warned = sum(c.status == "warn" for c in checks)
    lines.append(f"\n{len(checks)} checks: {len(checks) - failed - warned} passed, "
                 f"{warned} warnings, {failed} failed")
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> list[Check]:
    try:
        async with get_sessionmaker()() as db:
            return await run_all(db, expected_episodes=args.expected_episodes,
                                 earliest=args.earliest, sample=args.sample)
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.validation", description=__doc__)
    parser.add_argument("--expected-episodes", type=int, default=None,
                        help="episodes the corpus should have produced (default: count "
                             "the transcripts under TRANSCRIPTS_LOCAL_PATH, if present)")
    parser.add_argument("--earliest", type=date.fromisoformat,
                        default=DEFAULT_MIN_PUBLISH_DATE,
                        help="earliest plausible publish date "
                             f"(default {DEFAULT_MIN_PUBLISH_DATE})")
    parser.add_argument("--sample", type=int, default=20,
                        help="chunks to look up through both indexes (default 20)")
    args = parser.parse_args(argv)

    checks = asyncio.run(_run(args))
    print(render(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
