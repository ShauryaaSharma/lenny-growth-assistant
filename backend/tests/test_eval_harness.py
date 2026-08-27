"""The eval harness's own scoring logic.

`summarize()` is a pure function over a list of QuestionResult -- tested here
with hand-built results so the arithmetic (rates, precision, latency
percentiles, pass/fail against the PRD's thresholds) is verified without
needing a database or a live model. The harness's actual retrieval-calling
path (`evaluate_question`, `run`) is exercised against real data by running
`python -m app.evals.run_eval` directly -- that's the whole point of this
tool, and scripting it with a fake retriever would defeat it.
"""

from __future__ import annotations

from app.evals.run_eval import (
    MAX_FALSE_GROUND_RATE,
    MIN_GROUNDED_RATE,
    QuestionResult,
    summarize,
)


def make_result(
    category="in_domain",
    expected=True,
    actual=True,
    guest=None,
    guest_matched=None,
    latency=100,
) -> QuestionResult:
    return QuestionResult(
        query="q",
        category=category,
        expected_grounded=expected,
        actual_grounded=actual,
        best_similarity=0.7,
        latency_ms=latency,
        top_guests=["Some Guest"],
        expect_guest=guest,
        guest_matched=guest_matched,
        correct=(expected == actual),
    )


class TestGroundedAnswerRate:
    def test_all_in_domain_grounded_is_100_percent(self):
        results = [make_result(actual=True) for _ in range(5)]
        summary = summarize(results)
        assert summary["grounded_answer_rate"] == 1.0

    def test_half_in_domain_grounded_is_50_percent(self):
        results = [make_result(actual=True), make_result(actual=False)]
        summary = summarize(results)
        assert summary["grounded_answer_rate"] == 0.5

    def test_below_floor_fails_the_run(self):
        # 1 of 5 grounded = 20%, well under the 80% floor from the PRD.
        results = [make_result(actual=(i == 0)) for i in range(5)]
        summary = summarize(results)
        assert summary["grounded_answer_rate"] == 0.2
        assert summary["passed"] is False
        assert any("Grounded Answer Rate" in f for f in summary["failures"])

    def test_at_or_above_floor_does_not_fail_on_this_metric(self):
        # exactly at MIN_GROUNDED_RATE
        n = 10
        n_grounded = int(MIN_GROUNDED_RATE * n)
        results = [make_result(actual=(i < n_grounded)) for i in range(n)]
        summary = summarize(results)
        assert not any("Grounded Answer Rate" in f for f in summary["failures"])


class TestFalseGroundRate:
    def test_no_out_of_domain_questions_ground_is_a_pass(self):
        results = [
            make_result(category="out_of_domain", expected=False, actual=False)
            for _ in range(5)
        ]
        summary = summarize(results)
        assert summary["false_ground_rate"] == 0.0
        assert not any("False-Ground" in f for f in summary["failures"])

    def test_any_out_of_domain_question_grounding_fails_the_run(self):
        """This is the guardrail that matters most: the PRD requires 0%, not
        'mostly 0%'. A single false-ground must fail the run."""
        results = [
            make_result(category="out_of_domain", expected=False, actual=False),
            make_result(category="out_of_domain", expected=False, actual=True),  # bad
        ]
        summary = summarize(results)
        assert summary["false_ground_rate"] == 0.5
        assert summary["passed"] is False
        assert any("False-Ground Rate" in f for f in summary["failures"])
        assert MAX_FALSE_GROUND_RATE == 0.0


class TestGuestMatchPrecision:
    def test_precision_only_counts_questions_with_an_expected_guest(self):
        results = [
            make_result(guest=None, guest_matched=None),  # not counted
            make_result(guest="Adam Fishman", guest_matched=True),
            make_result(guest="Elena Verna", guest_matched=False),
        ]
        summary = summarize(results)
        assert summary["guest_match_precision"] == 0.5

    def test_no_guest_expectations_reports_none_not_zero(self):
        """A guest-precision of 0.0 and 'no data' are different facts; the
        report must be able to tell them apart."""
        results = [make_result(guest=None, guest_matched=None) for _ in range(3)]
        summary = summarize(results)
        assert summary["guest_match_precision"] is None


class TestLatencyReporting:
    def test_p50_and_p95_and_max(self):
        results = [make_result(latency=lat) for lat in [10, 20, 30, 40, 100]]
        summary = summarize(results)
        assert summary["latency_ms"]["p50"] == 30
        assert summary["latency_ms"]["max"] == 100

    def test_empty_result_set_does_not_crash(self):
        summary = summarize([])
        assert summary["grounded_answer_rate"] is None
        assert summary["false_ground_rate"] is None
        assert summary["latency_ms"]["p50"] is None
        # No data is not a violated guardrail.
        assert summary["passed"] is True


class TestOverallPassFail:
    def test_passes_when_both_guardrails_are_met(self):
        results = [make_result(category="in_domain", actual=True) for _ in range(8)] + [
            make_result(category="out_of_domain", expected=False, actual=False)
            for _ in range(8)
        ]
        summary = summarize(results)
        assert summary["passed"] is True
        assert summary["failures"] == []

    def test_reports_both_failures_when_both_guardrails_are_violated(self):
        results = [make_result(category="in_domain", actual=False) for _ in range(8)] + [
            make_result(category="out_of_domain", expected=False, actual=True)
            for _ in range(8)
        ]
        summary = summarize(results)
        assert summary["passed"] is False
        assert len(summary["failures"]) == 2
