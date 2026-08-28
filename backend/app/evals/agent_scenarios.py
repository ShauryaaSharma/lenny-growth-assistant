"""Golden agent scenarios -- the counterpart to golden_set.py, one layer up.

`golden_set.py` + `run_eval.py` test retrieval: does `search()` ground the
right questions and refuse the right ones. They never touch the agent loop,
the tool-calling decision, or the model.

This tests the agent: given a real conversational message, does `run_agent()`
-- running the real `LLMProvider`, the real tool registry, the real
deterministic guards -- actually call the right tool, actually refuse when it
should, actually create an artifact when asked, and do it *once* rather than
redundantly. That last property is not academic: it is exactly the axis on
which the most expensive defect in this build (agent-transcripts/09) lived --
a correct final answer produced by an agent loop that had silently done the
work twice. Retrieval-only evaluation cannot see that; only running the real
loop can.

Every scenario here corresponds to a real, previously-observed failure mode
(see the `notes` field) or a basic routing guarantee the PRD depends on.
Scenarios marked `slow=True` invoke the Ship 30 essay pipeline, which is
several sequential model calls and the most expensive thing in this system --
skip them for a fast iteration loop with `--exclude-slow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioTurn:
    message: str
    # At least one of these tools must appear somewhere in this turn's tool
    # calls. Empty = no requirement (e.g. a trivial greeting).
    expect_tools_any_of: list[str] = field(default_factory=list)
    # None of these tools may appear in this turn's tool calls.
    forbidden_tools: list[str] = field(default_factory=list)
    # Per-tool call-count ceiling. This is the regression guard for
    # agent-transcripts/09: a tool legitimately called more times than this
    # in one turn indicates the agent redid work it had already finished.
    max_tool_calls: dict[str, int] = field(default_factory=dict)
    expect_grounded: bool | None = None  # None = don't check
    expect_artifact: bool | None = None  # None = don't check
    expect_refusal_language: bool = False
    # Substrings that must NOT appear in the reply text. Regression guard for
    # a live bug: "Hey" (trivial, no search offered) still got back a
    # fabricated quote formatted as "[1]" with an invented guest name --
    # the model's citation habit firing with nothing real to cite. Routing
    # checks alone (forbidden_tools) don't catch this, since no tool was
    # ever called; only the reply text shows the fabrication.
    forbidden_reply_patterns: list[str] = field(default_factory=list)


@dataclass
class AgentScenario:
    name: str
    category: str
    turns: list[ScenarioTurn]
    slow: bool = False
    notes: str = ""


# Substring markers for "the assistant declined." Loose by design -- a small
# local model does not phrase refusals identically every time, and this only
# needs to catch the shape of a refusal, not its exact wording.
REFUSAL_MARKERS = (
    # Singular subject ("the podcast doesn't cover...").
    "doesn't cover", "does not cover",
    # Plural subject ("the transcripts do not cover...") -- missed on the
    # first version of this list, which caused two genuinely correct live
    # refusals ("Lenny's Podcast transcripts do not cover the topic of
    # sourdough starter recipes") to be scored as failures. The agent was
    # right; the harness's own marker list was too narrow.
    "don't cover", "do not cover",
    "not covered", "don't have", "doesn't have", "do not have", "does not have",
    "no information", "not something", "outside the scope",
    "not covered in", "not something i can",
)


AGENT_SCENARIOS: list[AgentScenario] = [
    AgentScenario(
        name="greeting_no_tools",
        category="trivial",
        notes=(
            "A greeting must not trigger retrieval at all -- see is_trivial(). "
            "Also must not fabricate a citation-style reply -- see "
            "agent-transcripts/12 for the live 'Hey' -> fake [1] quote bug."
        ),
        turns=[
            ScenarioTurn(
                message="hi",
                forbidden_tools=["search_transcripts", "write_ship30_essay", "create_artifact"],
                expect_grounded=False,
                forbidden_reply_patterns=["[1]", "[2]"],
            )
        ],
    ),
    AgentScenario(
        name="thanks_no_tools",
        category="trivial",
        notes="A different trivial pattern than 'hi' -- checks is_trivial()'s boundary, not just its first branch.",
        turns=[
            ScenarioTurn(
                message="thanks!",
                forbidden_tools=["search_transcripts", "write_ship30_essay", "create_artifact"],
            )
        ],
    ),
    AgentScenario(
        name="grounded_question_onboarding",
        category="grounded_qa",
        notes="The baseline case: a real in-domain question must search, ground, and answer without an artifact.",
        turns=[
            ScenarioTurn(
                message="What does Adam Fishman say about onboarding?",
                expect_tools_any_of=["search_transcripts"],
                expect_grounded=True,
                forbidden_tools=["create_artifact", "write_ship30_essay"],
                max_tool_calls={"search_transcripts": 2},
            )
        ],
    ),
    AgentScenario(
        name="out_of_domain_refusal",
        category="refusal",
        notes="The PRD's guardrail at the agent level, not just the retriever level: an out-of-domain question "
        "must be searched (so the model has a chance to be honest about it) and then refused, not answered.",
        turns=[
            ScenarioTurn(
                message="What's the best sourdough starter recipe?",
                expect_tools_any_of=["search_transcripts"],
                expect_grounded=False,
                expect_refusal_language=True,
                forbidden_tools=["create_artifact", "write_ship30_essay"],
            )
        ],
    ),
    AgentScenario(
        name="document_request_creates_artifact",
        category="artifact",
        notes="Regression guard for agent-transcripts/07: the model must render a requested document as an "
        "artifact, not as chat prose, and must do it exactly once.",
        turns=[
            ScenarioTurn(
                message="Build me a one-page onboarding audit checklist",
                expect_artifact=True,
                max_tool_calls={"create_artifact": 1},
            )
        ],
    ),
    AgentScenario(
        name="ungrounded_document_request_stays_refused",
        category="artifact_refusal",
        notes="Regression guard: the forced-artifact guard must never coerce a document out of an honest "
        "refusal. A corpus-unsupported document request should refuse, not produce a fabricated artifact.",
        turns=[
            ScenarioTurn(
                message="Give me a one-page sourdough starter checklist",
                expect_artifact=False,
                expect_refusal_language=True,
            )
        ],
    ),
    AgentScenario(
        name="two_turn_followup_context",
        category="multi_turn",
        notes="Checks that a pronoun follow-up in the second turn still resolves to a self-contained search "
        "query and still grounds -- session context must actually carry across turns.",
        turns=[
            ScenarioTurn(
                message="What does Adam Fishman say about onboarding?",
                expect_tools_any_of=["search_transcripts"],
                expect_grounded=True,
            ),
            ScenarioTurn(
                message="What else did he say that's useful for growth teams?",
                expect_tools_any_of=["search_transcripts"],
                expect_grounded=True,
            ),
        ],
    ),
    AgentScenario(
        name="essay_single_generation_no_redundant_calls",
        category="essay",
        slow=True,
        notes="Direct regression guard for agent-transcripts/09, the most expensive defect in this build: a "
        "successful essay generation must never trigger a second one. max_tool_calls={'write_ship30_essay': 1} "
        "is the exact property that bug violated.",
        turns=[
            ScenarioTurn(
                message="Write a Ship 30 essay about retention as a growth lever",
                expect_tools_any_of=["write_ship30_essay"],
                expect_artifact=True,
                max_tool_calls={"write_ship30_essay": 1},
            )
        ],
    ),
]


def by_category(category: str) -> list[AgentScenario]:
    return [s for s in AGENT_SCENARIOS if s.category == category]
