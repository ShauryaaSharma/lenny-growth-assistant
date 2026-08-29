# 13 — Small talk beyond the greeting allowlist ran a full retrieval and either fabricated or stiffly refused

**Context:** manually smoke-testing the live app in the browser while
preparing to record the demo video, sending ordinary conversational
messages ("Hey", "Nothing much.", "What you doin?") in sequence, the way an
actual user opening the app would.

## What happened

```
Hey
How's it going?
llama3.2:3b · 15.7s

Hey
Nothing much.
llama3.2:3b · 16.7s

What you doin?
Lenny's Podcast transcripts do not cover this topic. They do cover topics
like product management, growth, and company building. Would you like to
ask about something related to those topics?
llama3.2:3b · 30.7s
```

The first two are fine (routing and reply both reasonable). The third is
not: "What you doin?" is small talk, but it was routed as a *substantive*
question, triggered a full corpus search, found nothing above the grounding
floor, and answered with a stiff domain refusal after 30 seconds -- exactly
the failure mode a real user would read as the assistant being broken,
not careful.

## Root cause

`is_trivial()` (`app/agent/prompts.py`) matched only a fixed 12-phrase
allowlist (`TRIVIAL_PATTERNS`: "hi", "hey", "hello", "thanks", ...) plus a
length-3 shortcut. Anything conversational outside that list -- "what you
doin", "whats up", "how are you", "nothing much" -- fell through to the
substantive-question path by default, forcing retrieval on messages that
have no product/growth content to search for.

```python
>>> is_trivial("What you doin?")
False
>>> is_trivial("whats up")
False
>>> is_trivial("how are you")
False
```

This is a routing gap, not a model failure: the guard that exists
(`needs_grounding = not is_trivial(user_message)`, `app/agent/runtime.py`)
is correct once `is_trivial()` says yes -- the allowlist itself was just too
narrow to say yes often enough.

## The fix

Added `SMALL_TALK_EXACT` to `prompts.py`: a second, larger set of
conversational phrases, matched as **exact whole-message equality only**,
never as a prefix. That distinction matters -- `TRIVIAL_PATTERNS` is
prefix-matched ("thanks a lot" should still count as trivial), but doing the
same for `SMALL_TALK_EXACT` would silently skip retrieval on real questions
that happen to *open* the same way: "how are you going to measure
retention?" or "what you doing about churn in B2B SaaS?" both need to
search, and both start with a small-talk phrase. Exact-match-only avoids
that:

```python
>>> is_trivial("what you doin")          # small talk
True
>>> is_trivial("how are you going to measure retention?")  # real question
False
```

Verified live after the fix: "What you doin?" now returns "Just waiting for
your question about product management or growth. Go ahead and ask away!"
-- `grounded: false`, `tool_calls: []`, no retrieval run, no refusal
language.

Regression tests added to `tests/test_agent_routing.py::TestTrivialDetection`
covering both the small-talk phrases that must skip retrieval and the
questions that merely begin the same way but must still retrieve --
protecting against re-narrowing this fix back into the original bug.
