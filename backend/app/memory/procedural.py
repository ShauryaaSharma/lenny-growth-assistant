"""Procedural memory — operating principles, not facts.

Following the standard memory taxonomy (procedural / semantic / episodic):
procedural memory is "how to act" — durable instructions for behavior, as
opposed to facts about the world (semantic) or a record of what happened
(episodic, which for this system is the `sessions`/`messages` tables in
Postgres — see the module docstring in `reducers.py`).

This is data, not a prompt buried in Python, for the same reason
`skills/ship30/principles.md` is: a reviewer can read exactly what governing
principle the agent operates under without reading code, and changing it is a
documentation edit.

The first principle below is the one that actually matters most for this
product. It is not an arbitrary style choice -- it is a direct restatement of
what the brief itself asks for, in Oogway Labs' own language:

    "The strongest submissions will not only work technically; they will show
    clear judgment about what to build, what to simplify, how to communicate
    trade-offs, and how another team could run and extend the solution."

A Forward Deployed Engineer's product earns a client's trust by being honest
about its limits, not by being impressive. That is a different design target
than "answer everything fluently" -- it is the reason this assistant has a
grounding floor, an honest-refusal guard, and two eval harnesses that measure
whether it is actually keeping that promise, rather than trusting that it
does. That is procedural memory worth actually enforcing, not aspirational
copy -- see PROCEDURAL_MEMORY[0] below, and its enforcement points in
`agent/runtime.py` (the forced-retrieval and ungrounded guards) and
`app/evals/` (the harnesses that measure whether it holds).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Principle:
    name: str
    statement: str
    enforced_by: str


PROCEDURAL_MEMORY: tuple[Principle, ...] = (
    Principle(
        name="grounding_over_fluency",
        statement=(
            "An answer the user can verify beats an answer that merely sounds "
            "right. When the evidence doesn't support a claim, say so plainly "
            "instead of producing a fluent guess -- a confident wrong answer is "
            "worse than an honest 'I don't know', because it costs trust that "
            "doesn't come back."
        ),
        enforced_by=(
            "agent/runtime.py forced-retrieval + ungrounded guards; "
            "app/evals/run_eval.py measures the False-Ground Rate directly"
        ),
    ),
    Principle(
        name="small_surface_area",
        statement=(
            "Fewer, well-chosen tools beat a large capable-looking toolbox. "
            "Every additional tool is another way a small model can route "
            "incorrectly, so the registry stays deliberately minimal."
        ),
        enforced_by="agent/tools.py TOOL_SPECS (3 tools, by design)",
    ),
    Principle(
        name="verify_before_trusting",
        statement=(
            "A feature is not done because the code compiles and the tests "
            "pass. It is done once it has been run for real, against the real "
            "required model, and the actual output has been read -- not "
            "assumed. Every guard in this agent loop exists because something "
            "that looked correct on paper was checked and found wrong."
        ),
        enforced_by=(
            "app/evals/run_agent_eval.py; agent-transcripts/07-11 document each "
            "case this caught"
        ),
    ),
    Principle(
        name="operable_over_clever",
        statement=(
            "A solution a client's own engineers can run, diagnose, and extend "
            "without the original author in the room is worth more than one "
            "that is merely elegant. Prefer one command to start it, a typed "
            "error over a stack trace, and a comment that explains why over "
            "code that assumes the reader already knows."
        ),
        enforced_by=(
            "docker compose up as the only startup step; the typed error "
            "envelope in main.py; /health/deep"
        ),
    ),
)


def render_for_system_prompt() -> str:
    """Format the full set as a block -- for documentation and review, not for
    injection into the live prompt. Every existing rule in `SYSTEM_PROMPT` is a
    short imperative sentence, tuned for a 3B model that (per this project's
    own repeated finding) follows terse lists far more reliably than prose;
    stacking four abstract paragraphs on top of that list would dilute
    compliance with the rules that already do the enforcing. Use
    `render_primary_for_system_prompt()` for what actually goes live."""
    lines = ["Operating principles:"]
    for p in PROCEDURAL_MEMORY:
        lines.append(f"- {p.statement}")
    return "\n".join(lines)


def render_primary_for_system_prompt() -> str:
    """The one principle worth spending prompt budget on, restated as a single
    terse imperative sentence in the same style as `SYSTEM_PROMPT`'s numbered
    rules. The other three principles in PROCEDURAL_MEMORY are architectural
    (small tool surface, verify-before-trusting, operability) and are already
    enforced in code, not by asking the model nicely -- see each principle's
    `enforced_by` field."""
    return (
        "Procedural memory: a verified 'I don't know' is worth more than a "
        "fluent guess -- it keeps the trust that a wrong-but-confident answer "
        "would spend."
    )
