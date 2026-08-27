"""Ship 30 essay rubric and revision logic.

`check_rubric` and `_revision_prompt` are pure functions -- no model, no DB --
so they are tested directly here rather than only indirectly through the
agent-routing tests, which mock `write_ship30_essay` entirely and therefore
never exercise this file's actual logic.
"""

from __future__ import annotations

from app.skills.ship30.skill import (
    MAX_WORDS,
    MIN_WORDS,
    _revision_prompt,
    check_rubric,
)

GOOD_ESSAY = (
    "# How Onboarding Quietly Decides Your Retention Curve\n\n"
    + "Most teams treat onboarding as a checkbox. "
    + "word " * 250
    + "\n\n## Onboarding is the only universal touchpoint\n\n"
    + "word " * 250
    + " [1] "
    + "word " * 250
    + "\n\n## Retention is a habit-formation problem\n\n"
    + "word " * 250
    + " [2] "
    + "word " * 100
    + "\n\n- Map the first session\n- Cut every step that isn't value\n- Measure day-7, not day-1\n\n"
    + "## The specific fix\n\n"
    + "word " * 150
    + " [3] "
    + "This week, cut your onboarding flow by one step and measure the shift.\n"
)


class TestCheckRubric:
    def test_a_well_formed_essay_passes(self):
        report = check_rubric(GOOD_ESSAY, evidence_count=5)
        assert report["passed"] is True
        assert report["citation_count"] == 3

    def test_literal_n_placeholder_is_a_distinct_named_failure(self):
        """Regression: observed live on llama3.2:3b -- the model wrote the
        literal characters '[n]' instead of a real citation number, because an
        earlier version of the prompt used '[n]' as meta-notation for "insert
        a number here" and the model took it literally."""
        essay = GOOD_ESSAY.replace("[1]", "[n]").replace("[2]", "[n]").replace("[3]", "[n]")
        report = check_rubric(essay, evidence_count=5)
        assert report["checks"]["no_literal_placeholder_citations"] is False
        assert report["passed"] is False
        # The literal string must not be miscounted as a real citation.
        assert report["citation_count"] == 0

    def test_real_citations_are_not_flagged_as_placeholders(self):
        report = check_rubric(GOOD_ESSAY, evidence_count=5)
        assert report["checks"]["no_literal_placeholder_citations"] is True

    def test_word_count_too_short_fails(self):
        short_essay = "# Title\n\nOnly a few words here, nowhere near the target length.\n"
        report = check_rubric(short_essay, evidence_count=3)
        assert report["checks"]["word_count_in_range"] is False
        assert report["word_count"] < MIN_WORDS

    def test_word_count_too_long_fails(self):
        long_essay = "# Title\n\n" + "word " * (MAX_WORDS + 500)
        report = check_rubric(long_essay, evidence_count=3)
        assert report["checks"]["word_count_in_range"] is False
        assert report["word_count"] > MAX_WORDS

    def test_citation_outside_evidence_range_is_flagged(self):
        essay = GOOD_ESSAY.replace("[3]", "[99]")
        report = check_rubric(essay, evidence_count=5)
        assert report["checks"]["citations_in_range"] is False
        assert 99 in report["invalid_citations"]

    def test_too_few_sections_fails(self):
        essay = "# Title\n\n" + "word " * 300 + "\n\n## Only one section\n\n" + "word " * 300
        report = check_rubric(essay, evidence_count=3)
        assert report["checks"]["section_count_3_to_5"] is False

    def test_missing_list_fails(self):
        no_list_essay = GOOD_ESSAY.replace("- Map the first session\n", "").replace(
            "- Cut every step that isn't value\n", ""
        ).replace("- Measure day-7, not day-1\n", "")
        report = check_rubric(no_list_essay, evidence_count=5)
        assert report["checks"]["has_list"] is False

    def test_excessive_bold_fails(self):
        essay = "# Title\n\n" + "**bold** word " * 20 + "\n\n## Section\n\n[1] " + "word " * 300
        report = check_rubric(essay, evidence_count=3)
        assert report["checks"]["bold_not_excessive"] is False


class TestRevisionPrompt:
    def test_names_the_literal_placeholder_defect_explicitly(self):
        """The revision instruction must tell the model exactly what to fix --
        a vague 'add more citations' message is what let this defect survive
        one revision pass in the first place."""
        essay_with_bug = GOOD_ESSAY.replace("[1]", "[n]").replace("[2]", "[n]").replace(
            "[3]", "[n]"
        )
        report = check_rubric(essay_with_bug, evidence_count=5)
        messages = _revision_prompt(essay_with_bug, report, evidence="evidence block")
        instruction_text = messages[-1].content
        assert "[n]" in instruction_text
        assert "literal" in instruction_text.lower()

    def test_does_not_mention_checks_that_already_passed(self):
        report = check_rubric(GOOD_ESSAY, evidence_count=5)
        assert report["passed"] is True  # nothing to revise
