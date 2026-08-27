"""The golden evaluation set.

This is the part a test suite cannot give you: a fixed, labeled sample of real
questions, run against the real retriever, scored against an expected outcome.
Unit tests (`tests/test_agent_routing.py`, `tests/test_retriever.py`) verify
that the *code* behaves correctly given a scripted input. This verifies that
the *system* behaves correctly given a real one -- which is what the PRD's
"Grounded Answer Rate >= 80%" and "0% of out-of-domain questions answered
without evidence" success metrics actually require someone to measure.

Every `EXPECT_GROUNDED` question below names a guest confirmed present in the
ingested corpus at the time this set was written (spot-checked directly
against the `episodes` table) -- a question about content that legitimately
has not been ingested yet is not a retrieval failure, and would only pollute
the signal this harness exists to produce. If you add episodes or run this
against a fresh/partial corpus, check `python -m app.evals.run_eval --list`
against `SELECT guest FROM episodes` first.

`EXPECT_UNGROUNDED` questions are deliberately unrelated to product, growth,
or company-building -- topics no episode of Lenny's Podcast would plausibly
cover. A harness that only tests the easy in-domain case would miss the
guardrail that actually matters most: the assistant's willingness to say it
doesn't know.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenQuestion:
    query: str
    expect_grounded: bool
    category: str
    # For grounded questions only: a guest whose episode should plausibly
    # appear in the top results. Checked as a soft precision signal, not a
    # hard pass/fail -- hybrid retrieval may legitimately surface a different
    # equally-relevant guest for a broad question.
    expect_guest: str | None = None
    notes: str = ""


GOLDEN_SET: list[GoldenQuestion] = [
    # ---- In-domain: specific to one confirmed-ingested guest ----
    GoldenQuestion(
        "What does Adam Fishman say about onboarding?",
        expect_grounded=True, category="in_domain", expect_guest="Adam Fishman",
    ),
    GoldenQuestion(
        "How do you build a high-performing growth team?",
        expect_grounded=True, category="in_domain", expect_guest="Adam Fishman",
    ),
    GoldenQuestion(
        "How do you develop a growth model for a marketplace business?",
        expect_grounded=True, category="in_domain", expect_guest="Dan Hockenmaier",
    ),
    GoldenQuestion(
        "How do I find hidden growth opportunities in my product?",
        expect_grounded=True, category="in_domain", expect_guest="Albert Cheng",
    ),
    GoldenQuestion(
        "What growth tactics never actually work?",
        expect_grounded=True, category="in_domain", expect_guest="Elena Verna",
    ),
    GoldenQuestion(
        "Why will ChatGPT become a major growth channel?",
        expect_grounded=True, category="in_domain", expect_guest="Brian Balfour",
    ),
    GoldenQuestion(
        "What's the ultimate guide to product-led sales?",
        expect_grounded=True, category="in_domain", expect_guest="Elena Verna 3.0",
    ),
    # ---- In-domain: broad questions, no single guest expected ----
    GoldenQuestion(
        "How do you know when you've found product-market fit?",
        expect_grounded=True, category="in_domain",
        notes="Broad topic; many episodes plausibly discuss this.",
    ),
    GoldenQuestion(
        "When should a founder hire their first product manager?",
        expect_grounded=True, category="in_domain",
    ),
    GoldenQuestion(
        "Why does user retention matter more than acquisition for growth?",
        expect_grounded=True, category="in_domain",
    ),
    GoldenQuestion(
        "What makes a good onboarding experience for new users?",
        expect_grounded=True, category="in_domain",
    ),
    GoldenQuestion(
        "How should I think about pricing a B2B SaaS product?",
        expect_grounded=True, category="in_domain",
    ),
    GoldenQuestion(
        "What's the difference between growth hacking and sustainable growth?",
        expect_grounded=True, category="in_domain",
    ),
    GoldenQuestion(
        "How do I build a great company culture as a startup scales?",
        expect_grounded=True, category="in_domain",
    ),

    # ---- Out-of-domain: unrelated to product/growth/company-building ----
    GoldenQuestion(
        "What's the best sourdough starter recipe?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "How do I fix a flat tire on a road bike?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "What's the capital of Mongolia?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "Can you explain general relativity in simple terms?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "What's a good beginner workout routine for building muscle?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "How do I train my dog to stop barking at the mailman?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "What's a good recipe for vegan lasagna?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "What's the plot of the movie Inception?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "How do I change the oil in my car?",
        expect_grounded=False, category="out_of_domain",
    ),
    GoldenQuestion(
        "What's the healthiest way to cook salmon?",
        expect_grounded=False, category="out_of_domain",
    ),
]


def by_category(category: str) -> list[GoldenQuestion]:
    return [q for q in GOLDEN_SET if q.category == category]
