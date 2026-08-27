"""The Ship 30 for 30 essay skill.

This is a *skill*, not a prompt: the writing principles live in `principles.md`
as versioned data, and the generation is a deterministic multi-step pipeline
with a programmatic quality gate. Three properties follow from that:

  1. Changing the house style is a documentation edit, not a code change.
  2. A reviewer can read exactly what the model was instructed to do.
  3. The output is *checked* before it is returned. A one-shot prompt has no
     way to know it produced 400 words when it was asked for 1,250; this does,
     and revises once when the draft misses.

Pipeline: retrieve (wide) -> outline -> draft -> rubric check -> revise if needed.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.base import ChatMessage
from app.llm.registry import chat_with_fallback
from app.logging import get_logger
from app.rag.retriever import RetrievalResult, search

log = get_logger(__name__)

PRINCIPLES_PATH = Path(__file__).parent / "principles.md"

TARGET_WORDS = 1250
MIN_WORDS = 1050
MAX_WORDS = 1450
ESSAY_TOP_K = 14  # wider than conversational retrieval -- an essay needs more material


@lru_cache(maxsize=1)
def load_principles() -> str:
    return PRINCIPLES_PATH.read_text(encoding="utf-8")


def _outline_prompt(topic: str, evidence: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a writing strategist trained on the Ship 30 for 30 method. "
                "You plan essays; you do not write them yet.\n\n"
                f"{load_principles()}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"TOPIC: {topic}\n\n"
                f"EVIDENCE FROM LENNY'S PODCAST TRANSCRIPTS:\n{evidence}\n\n"
                "Produce a plan for a ~1,250-word essay, and nothing else:\n"
                "1. HEADLINE: one line, containing the audience, the topic, and a promise.\n"
                "2. HOOK: the literal first sentence, under 20 words, a concrete claim.\n"
                "3. SECTIONS: 3 to 4 section headings, each stating a claim rather than "
                "naming a topic. Under each, list which evidence numbers support it -- for "
                "example, write '1, 3' or '[1], [3]' using the actual numbers from the "
                "evidence list above. Never write the literal characters '[n]'.\n"
                "4. TAKEAWAY: the one specific action the reader should take this week.\n\n"
                "Use only the evidence provided. Do not write the essay."
            ),
        ),
    ]


def _draft_prompt(topic: str, evidence: str, outline: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a writer trained on the Ship 30 for 30 method. Follow these "
                "principles precisely.\n\n"
                f"{load_principles()}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"TOPIC: {topic}\n\n"
                f"APPROVED PLAN:\n{outline}\n\n"
                f"EVIDENCE FROM LENNY'S PODCAST TRANSCRIPTS:\n{evidence}\n\n"
                f"Write the full essay in Markdown, following the plan exactly.\n\n"
                f"Hard requirements:\n"
                f"- {MIN_WORDS}-{MAX_WORDS} words. This is the most common failure; "
                f"count as you go and develop each section fully.\n"
                "- Start with the headline as a `# ` heading.\n"
                "- Use `## ` for each section heading.\n"
                "- Include at least one bulleted or numbered list.\n"
                "- Bold the load-bearing sentence in each section, no more.\n"
                "- Cite evidence inline as [1], [2] matching the numbers above.\n"
                "- Attribute named ideas to the guest who said them.\n"
                "- End with the specific takeaway. Do not write a summary.\n"
                "- Output only the essay. No preamble, no commentary about the essay."
            ),
        ),
    ]


def check_rubric(essay: str, evidence_count: int) -> dict:
    """Programmatic pass over the rubric in principles.md.

    Only mechanically checkable rules live here. Judgement calls (is the hook
    actually compelling?) are left to the prompt, because a regex that pretends
    to measure them would be worse than not measuring them.
    """
    words = len(essay.split())
    headings = re.findall(r"^##\s+(.+)$", essay, re.MULTILINE)
    h1 = re.findall(r"^#\s+(.+)$", essay, re.MULTILINE)
    citations = {int(n) for n in re.findall(r"\[(\d{1,2})\]", essay)}
    bold_count = len(re.findall(r"\*\*[^*]+\*\*", essay))
    has_list = bool(re.search(r"^\s*([-*]|\d+\.)\s+", essay, re.MULTILINE))
    out_of_range = {c for c in citations if c < 1 or c > evidence_count}
    # A small model can mistake the outline/revision prompts' meta-notation for
    # literal output and write "[n]" verbatim instead of a real number -- caught
    # live on llama3.2:3b. Checked separately from citation count so the
    # revision instruction can name the exact defect rather than a vague count.
    has_literal_n_placeholder = bool(re.search(r"\[n\]", essay, re.IGNORECASE))

    first_sentence = ""
    body = re.sub(r"^#.*$", "", essay, count=1, flags=re.MULTILINE).strip()
    if body:
        first_sentence = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]

    checks = {
        "word_count_in_range": MIN_WORDS <= words <= MAX_WORDS,
        "has_headline": bool(h1),
        "hook_is_short": 0 < len(first_sentence.split()) <= 25,
        "section_count_3_to_5": 3 <= len(headings) <= 5,
        "has_list": has_list,
        "bold_not_excessive": bold_count <= 12,
        "has_three_citations": len(citations) >= 3,
        "citations_in_range": not out_of_range,
        "no_literal_placeholder_citations": not has_literal_n_placeholder,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "word_count": words,
        "section_count": len(headings),
        "citation_count": len(citations),
        "bold_count": bold_count,
        "invalid_citations": sorted(out_of_range),
    }


def _revision_prompt(essay: str, report: dict, evidence: str) -> list[ChatMessage]:
    failures = [name for name, ok in report["checks"].items() if not ok]
    instructions: list[str] = []
    if "word_count_in_range" in failures:
        words = report["word_count"]
        if words < MIN_WORDS:
            instructions.append(
                f"The draft is {words} words; it must be {MIN_WORDS}-{MAX_WORDS}. "
                f"Expand the existing sections with more evidence and concrete detail. "
                f"Do NOT add new sections and do NOT pad with restatement."
            )
        else:
            instructions.append(
                f"The draft is {words} words; it must be at most {MAX_WORDS}. "
                f"Cut restatement and hedging. Keep every section."
            )
    if "section_count_3_to_5" in failures:
        instructions.append(
            f"There are {report['section_count']} `## ` sections; there must be 3 to 5."
        )
    if "no_literal_placeholder_citations" in failures:
        instructions.append(
            "The essay contains the literal characters '[n]' where a real citation "
            "number belongs. Replace every '[n]' with the actual number of the "
            "evidence excerpt it refers to -- for example [1] or [3]. Never leave "
            "'[n]' in the output."
        )
    if "has_three_citations" in failures:
        instructions.append(
            f"Only {report['citation_count']} distinct citations appear, using real "
            f"bracketed numbers from the evidence list -- for example [1] or [3]. "
            f"Cite at least 3 distinct pieces of evidence this way."
        )
    if "citations_in_range" in failures:
        instructions.append(
            f"These citations do not exist: {report['invalid_citations']}. "
            f"Remove them or replace them with valid evidence numbers."
        )
    if "has_list" in failures:
        instructions.append("Add at least one bulleted or numbered list.")
    if "bold_not_excessive" in failures:
        instructions.append(
            f"Bold is used {report['bold_count']} times. Reduce to at most 12 uses."
        )
    if "hook_is_short" in failures:
        instructions.append("Rewrite the opening sentence to be under 20 words and concrete.")
    if "has_headline" in failures:
        instructions.append("Add a `# ` headline as the first line.")

    return [
        ChatMessage(
            role="system",
            content=(
                "You are revising an essay to meet its specification. Preserve the voice, "
                "argument, and structure. Change only what the instructions require.\n\n"
                f"{load_principles()}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"EVIDENCE:\n{evidence}\n\n"
                f"CURRENT DRAFT:\n{essay}\n\n"
                "REQUIRED FIXES:\n- " + "\n- ".join(instructions) + "\n\n"
                "Return only the revised essay in Markdown."
            ),
        ),
    ]


async def write_ship30_essay(db: AsyncSession, topic: str) -> dict:
    """Generate a Ship 30-style essay grounded in the transcript corpus."""
    started = time.perf_counter()

    retrieval: RetrievalResult = await search(db, topic, top_k=ESSAY_TOP_K)
    if not retrieval.grounded:
        log.info("ship30_ungrounded", topic=topic[:80], reason=retrieval.reason)
        return {
            "ok": False,
            "reason": "insufficient_evidence",
            "message": (
                "The transcript corpus does not contain enough material on this topic "
                "to write a grounded essay. Try a topic closer to product, growth, "
                "retention, pricing, hiring, or company building."
            ),
            "citations": [],
        }

    evidence = retrieval.as_context_block(max_chars=20000)
    evidence_count = len(retrieval.chunks)

    outline_resp = await chat_with_fallback(_outline_prompt(topic, evidence), temperature=0.5)
    outline = outline_resp.content
    log.info("ship30_outline_done", chars=len(outline), latency_ms=outline_resp.latency_ms)

    draft_resp = await chat_with_fallback(
        _draft_prompt(topic, evidence, outline), temperature=0.7, max_tokens=3000
    )
    essay = draft_resp.content.strip()

    report = check_rubric(essay, evidence_count)
    revised = False
    if not report["passed"]:
        log.info(
            "ship30_revising",
            failures=[k for k, v in report["checks"].items() if not v],
            word_count=report["word_count"],
        )
        revision = await chat_with_fallback(
            _revision_prompt(essay, report, evidence), temperature=0.6, max_tokens=3000
        )
        candidate = revision.content.strip()
        candidate_report = check_rubric(candidate, evidence_count)
        # Keep the revision only if it is actually better -- small local models
        # sometimes return a shorter, worse draft when asked to expand.
        if sum(candidate_report["checks"].values()) >= sum(report["checks"].values()):
            essay, report, revised = candidate, candidate_report, True

    total_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "ship30_complete",
        word_count=report["word_count"],
        passed=report["passed"],
        revised=revised,
        latency_ms=total_ms,
    )

    return {
        "ok": True,
        "essay": essay,
        "outline": outline,
        "rubric": report,
        "revised": revised,
        "citations": [c.as_citation() for c in retrieval.chunks],
        "latency_ms": total_ms,
        "provider": draft_resp.provider,
        "model": draft_resp.model,
    }
