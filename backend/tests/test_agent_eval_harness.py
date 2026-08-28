"""The agent-scenario harness's own scoring logic.

`score_turn()` and `summarize()` are pure functions over a fabricated
AgentResult-like object -- tested here without a database, a model, or the
agent loop, mirroring `test_eval_harness.py`'s approach for the retrieval
harness. The harness's actual value is in calling the real `run_agent()`
against the real model (`python -m app.evals.run_agent_eval`); scripting that
away here would defeat the entire point of building it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.evals.agent_scenarios import AgentScenario, ScenarioTurn
from app.evals.run_agent_eval import ScenarioResult, TurnResult, run_scenario, score_turn, summarize


@dataclass
class FakeAgentResult:
    content: str = "some answer"
    tool_calls: list[dict] = field(default_factory=list)
    grounded: bool = True
    artifacts: list = field(default_factory=list)
    latency_ms: int = 100


def tool_call(name: str, ok: bool = True) -> dict:
    return {"tool": name, "ok": ok, "latency_ms": 10}


class TestExpectToolsAnyOf:
    def test_passes_when_expected_tool_was_called(self):
        turn = ScenarioTurn(message="m", expect_tools_any_of=["search_transcripts"])
        result = FakeAgentResult(tool_calls=[tool_call("search_transcripts")])
        scored = score_turn(turn, result)
        assert scored.checks["expected_tool_called"] is True
        assert scored.correct is True

    def test_fails_when_no_expected_tool_was_called(self):
        turn = ScenarioTurn(message="m", expect_tools_any_of=["search_transcripts"])
        result = FakeAgentResult(tool_calls=[])
        scored = score_turn(turn, result)
        assert scored.checks["expected_tool_called"] is False
        assert scored.correct is False

    def test_any_of_multiple_tools_satisfies_the_check(self):
        turn = ScenarioTurn(message="m", expect_tools_any_of=["write_ship30_essay", "create_artifact"])
        result = FakeAgentResult(tool_calls=[tool_call("create_artifact")])
        assert score_turn(turn, result).correct is True


class TestForbiddenTools:
    def test_passes_when_no_tools_called(self):
        turn = ScenarioTurn(
            message="hi",
            forbidden_tools=["search_transcripts", "write_ship30_essay", "create_artifact"],
        )
        result = FakeAgentResult(tool_calls=[])
        assert score_turn(turn, result).correct is True

    def test_fails_when_a_forbidden_tool_was_called(self):
        """This is the trivial-greeting regression: a "hi" that triggers a
        search is a real routing defect, not a false alarm."""
        turn = ScenarioTurn(message="hi", forbidden_tools=["search_transcripts"])
        result = FakeAgentResult(tool_calls=[tool_call("search_transcripts")])
        scored = score_turn(turn, result)
        assert scored.checks["no_forbidden_tools"] is False

    def test_a_blocked_forbidden_tool_attempt_does_not_count(self):
        """A forbidden tool the runtime itself blocked (ok=False) never
        actually ran -- the guard did its job, so this must not fail."""
        turn = ScenarioTurn(message="m", forbidden_tools=["create_artifact"])
        result = FakeAgentResult(tool_calls=[tool_call("create_artifact", ok=False)])
        assert score_turn(turn, result).correct is True


class TestRedundantToolCalls:
    def test_single_call_within_limit_passes(self):
        turn = ScenarioTurn(message="m", max_tool_calls={"write_ship30_essay": 1})
        result = FakeAgentResult(tool_calls=[tool_call("write_ship30_essay")])
        assert score_turn(turn, result).correct is True

    def test_a_second_call_exceeding_the_limit_fails(self):
        """The direct regression guard for agent-transcripts/09: a tool
        called twice in one turn when once was correct must fail, even
        though the final answer might still look fine."""
        turn = ScenarioTurn(message="m", max_tool_calls={"write_ship30_essay": 1})
        result = FakeAgentResult(
            tool_calls=[tool_call("write_ship30_essay"), tool_call("write_ship30_essay")]
        )
        scored = score_turn(turn, result)
        assert scored.checks["no_redundant_write_ship30_essay"] is False
        assert scored.correct is False

    def test_zero_calls_is_within_any_positive_limit(self):
        turn = ScenarioTurn(message="m", max_tool_calls={"search_transcripts": 2})
        result = FakeAgentResult(tool_calls=[])
        assert score_turn(turn, result).correct is True

    def test_a_blocked_second_attempt_does_not_count_as_redundant(self):
        """The runtime's own guard blocking a second create_artifact call
        (ok=False) is the safety net working correctly, not a defect --
        counting it toward the limit would score correct behaviour as a
        failure. Observed live: exactly this pattern
        (tools=['create_artifact', 'search_transcripts', 'create_artifact'])
        with the second call blocked, which the first version of this scorer
        incorrectly failed."""
        turn = ScenarioTurn(message="m", max_tool_calls={"create_artifact": 1})
        result = FakeAgentResult(
            tool_calls=[
                tool_call("create_artifact", ok=True),
                tool_call("search_transcripts", ok=True),
                tool_call("create_artifact", ok=False),  # blocked by the runtime's own guard
            ]
        )
        scored = score_turn(turn, result)
        assert scored.checks["no_redundant_create_artifact"] is True
        assert scored.correct is True

    def test_two_actually_executed_calls_still_fail(self):
        """The filter must not become a loophole: two genuinely successful
        calls to the same tool is still the real defect this check exists
        to catch."""
        turn = ScenarioTurn(message="m", max_tool_calls={"create_artifact": 1})
        result = FakeAgentResult(
            tool_calls=[tool_call("create_artifact", ok=True), tool_call("create_artifact", ok=True)]
        )
        assert score_turn(turn, result).correct is False


class TestGroundedExpectation:
    def test_matches_expected_true(self):
        turn = ScenarioTurn(message="m", expect_grounded=True)
        assert score_turn(turn, FakeAgentResult(grounded=True)).correct is True

    def test_mismatch_fails(self):
        turn = ScenarioTurn(message="m", expect_grounded=True)
        scored = score_turn(turn, FakeAgentResult(grounded=False))
        assert scored.checks["grounded_as_expected"] is False
        assert scored.correct is False

    def test_none_means_not_checked(self):
        turn = ScenarioTurn(message="m", expect_grounded=None)
        scored = score_turn(turn, FakeAgentResult(grounded=False))
        assert "grounded_as_expected" not in scored.checks
        assert scored.correct is True


class TestArtifactExpectation:
    def test_expected_artifact_present_passes(self):
        turn = ScenarioTurn(message="m", expect_artifact=True)
        result = FakeAgentResult(artifacts=[object()])
        assert score_turn(turn, result).correct is True

    def test_expected_no_artifact_but_one_was_created_fails(self):
        """Regression guard: an ungrounded document request must not produce
        an artifact even if the model tried to create one anyway."""
        turn = ScenarioTurn(message="m", expect_artifact=False)
        result = FakeAgentResult(artifacts=[object()])
        scored = score_turn(turn, result)
        assert scored.checks["artifact_as_expected"] is False
        assert scored.correct is False


class TestRefusalLanguage:
    def test_recognises_a_refusal_phrase(self):
        turn = ScenarioTurn(message="m", expect_refusal_language=True)
        result = FakeAgentResult(content="Lenny's Podcast doesn't cover sourdough baking.")
        assert score_turn(turn, result).correct is True

    def test_an_answer_with_no_refusal_language_fails(self):
        turn = ScenarioTurn(message="m", expect_refusal_language=True)
        result = FakeAgentResult(content="Here is a detailed sourdough recipe: ...")
        scored = score_turn(turn, result)
        assert scored.checks["refusal_language_present"] is False
        assert scored.correct is False


class TestForbiddenReplyPatterns:
    def test_a_clean_reply_passes(self):
        turn = ScenarioTurn(message="hi", forbidden_reply_patterns=["[1]", "[2]"])
        result = FakeAgentResult(content="Hey, how's it going?")
        scored = score_turn(turn, result)
        assert scored.checks["no_forbidden_reply_patterns"] is True
        assert scored.correct is True

    def test_a_fabricated_citation_in_an_ungrounded_reply_fails(self):
        """Regression guard: 'Hey' once got back a fabricated quote formatted
        as [1] with an invented guest name, even though no search ran --
        see agent-transcripts/12."""
        turn = ScenarioTurn(message="hi", forbidden_reply_patterns=["[1]", "[2]"])
        result = FakeAgentResult(content='"I made mistakes" - Lenny Russell [1]')
        scored = score_turn(turn, result)
        assert scored.checks["no_forbidden_reply_patterns"] is False
        assert scored.correct is False


class TestTurnWithNoExpectations:
    def test_a_turn_with_no_checks_configured_is_trivially_correct(self):
        turn = ScenarioTurn(message="m")
        scored = score_turn(turn, FakeAgentResult())
        assert scored.checks == {}
        assert scored.correct is True


class TestSummarize:
    def _scenario_result(self, name, category, correct, notes=""):
        turn = TurnResult(
            message="m", checks={"x": correct}, correct=correct,
            tool_calls=[], grounded=True, has_artifact=False,
            content_snippet="", latency_ms=10,
        )
        return ScenarioResult(
            name=name, category=category, slow=False, turns=[turn], correct=correct, notes=notes
        )

    def test_all_passing_scenarios_pass_overall(self):
        results = [self._scenario_result("a", "cat1", True), self._scenario_result("b", "cat1", True)]
        summary = summarize(results)
        assert summary["passed"] is True
        assert summary["failures"] == []

    def test_one_failing_scenario_fails_overall_and_is_named(self):
        results = [self._scenario_result("a", "cat1", True), self._scenario_result("b", "cat2", False)]
        summary = summarize(results)
        assert summary["passed"] is False
        assert any("b:" in f for f in summary["failures"])

    def test_category_pass_rate_is_computed_per_category(self):
        results = [
            self._scenario_result("a", "cat1", True),
            self._scenario_result("b", "cat1", False),
            self._scenario_result("c", "cat2", True),
        ]
        summary = summarize(results)
        assert summary["category_pass_rate"]["cat1"] == 0.5
        assert summary["category_pass_rate"]["cat2"] == 1.0

    def test_empty_results_do_not_report_a_pass(self):
        """No scenarios run is not the same as all scenarios passing -- an
        empty run must not silently report green."""
        summary = summarize([])
        assert summary["passed"] is False


class TestRunScenarioSurvivesProviderFailure:
    """A provider timeout on one scenario -- a real risk on a CPU-only local
    model under load, observed live when this harness first ran concurrently
    with ingestion -- must be recorded as a failed turn, not crash the whole
    suite and lose every other scenario's result."""

    async def test_llm_error_is_caught_and_recorded_not_raised(self, monkeypatch):
        from app.llm.base import LLMTimeoutError

        async def failing_run_agent(db, session_id, message, history):
            raise LLMTimeoutError("Ollama did not respond within 180s.")

        monkeypatch.setattr("app.evals.run_agent_eval.run_agent", failing_run_agent)
        monkeypatch.setattr(
            "app.evals.run_agent_eval.get_sessionmaker", lambda: _NullSessionmaker()
        )

        scenario = AgentScenario(
            name="x", category="cat", turns=[ScenarioTurn(message="hi")]
        )
        result = await run_scenario(scenario)  # must not raise

        assert result.correct is False
        assert len(result.turns) == 1
        assert result.turns[0].checks["provider_call_succeeded"] is False


class _NullSessionmaker:
    """Stand-in for get_sessionmaker() that needs no real database -- the
    failing run_agent above never actually touches `db`."""

    def __call__(self):
        return self

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False
