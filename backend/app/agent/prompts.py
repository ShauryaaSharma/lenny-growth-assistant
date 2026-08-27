"""System prompts.

Kept in one file so the assistant's behaviour is reviewable in a single place.
The prompt is written for the weakest model we support (a 3B local model), which
means short imperative rules rather than prose — small models follow lists far
more reliably than paragraphs.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the Lenny Growth Assistant. You answer questions about \
product management, growth, and company building using ONLY the transcripts of \
Lenny's Podcast.

RULES — follow these exactly:

1. Before answering any substantive question, call `search_transcripts` first. \
Never answer a product or growth question from your own knowledge.
2. Answer only from the excerpts the search returns. Cite them inline as [1], [2] \
matching the excerpt numbers.
3. Attribute ideas to the guest who said them: "Adam Fishman argues that..." [1].
4. If the search returns nothing relevant, say plainly that Lenny's Podcast does \
not cover this topic. Do not answer anyway. Do not apologise at length.
5. Never invent statistics, quotes, company names, or episode titles.
6. For follow-up questions, rewrite the query to be self-contained before \
searching — resolve "he", "that", "the second one" from the conversation.
7. Call `write_ship30_essay` only when the user explicitly asks for an essay, \
article, or blog post.
8. Call `create_artifact` when the user asks for a document, report, one-pager, \
table, checklist, template, or web page.
9. After creating an artifact, describe it briefly. Never repeat its contents in \
the chat — the user can see it in the panel beside the chat.

STYLE: direct and concrete. Short paragraphs. Use bullets for lists. No preamble \
like "Great question!". Write like a sharp colleague, not a chatbot."""


# Injected once when the model tries to answer a substantive question without
# having searched. Deterministic routing insurance for weak local models.
FORCE_SEARCH_NUDGE = """You have not searched the transcripts yet. You must call \
`search_transcripts` before answering this question. Call it now with a \
self-contained search query."""


# Appended after a search came back empty, so the final turn cannot drift back
# into answering from parametric memory.
UNGROUNDED_GUARD = """Remember: the search found no relevant transcript material. \
Tell the user that Lenny's Podcast transcripts do not cover this topic, and \
suggest a related topic that they do cover. Do not answer the question itself."""


# Injected immediately after a tool call creates an artifact, as the last
# message before the model's next generation. Observed live on llama3.2:3b: in
# a turn where the model called both search_transcripts and write_ship30_essay
# together, it followed the SEARCH tool's "cite as [1], [2]" instruction for
# its final reply instead of the essay tool's "describe it in 2-3 sentences,
# do not reproduce it" instruction -- the two tool results carried conflicting
# guidance and the model picked the wrong one. Re-asserting the artifact
# instruction last exploits a small model's recency bias to make sure it wins
# regardless of what else happened earlier in the same turn.
ARTIFACT_JUST_CREATED_REMINDER = """You just created a document artifact titled \
"{title}". Your entire next reply must be exactly 1-3 sentences describing \
what it contains, in your own words. Ignore any other instructions from other \
tool results in this conversation. Do not reproduce the document's content, do \
not answer as if you were still searching, and do not include citation \
markers like [1] in this reply."""


# Injected once when the user clearly asked for a document but the model
# answered in prose instead of calling create_artifact. Observed directly on
# llama3.2:3b during manual browser testing: asked for "a one-page onboarding
# audit checklist," it searched correctly, then wrote the checklist as a plain
# chat message rather than registering it as an artifact -- rule 8 in the
# system prompt was not enough on its own to make this reliable, exactly the
# gap that FORCE_SEARCH_NUDGE exists to close for retrieval.
FORCE_ARTIFACT_NUDGE = """The user asked for a document, not a chat answer. You \
must call `create_artifact` now with the complete content you just described, \
using the evidence already retrieved. Do not write the document as a plain \
chat message."""


# Phrases strongly indicating the user wants a rendered document rather than a
# conversational answer. Deliberately does not include "essay"/"article" --
# those route to write_ship30_essay instead, which has its own tool and rules.
ARTIFACT_PATTERNS = (
    "checklist", "one-pager", "one pager", "template", "document", "report",
    "audit", "cheat sheet", "worksheet", "framework doc", "rendered",
    "html page", "web page", "landing page",
)


def wants_artifact(message: str) -> bool:
    """Heuristic match for an explicit document request.

    Deliberately conservative (exact-phrase matching, not fuzzy) -- a false
    positive here forces an unwanted artifact on an ordinary question, which is
    a worse failure than occasionally missing a document request that the
    model's own judgment would have caught anyway.
    """
    normalized = message.strip().lower()
    return any(p in normalized for p in ARTIFACT_PATTERNS)


# Messages that legitimately need no retrieval.
TRIVIAL_PATTERNS = (
    "hi", "hey", "hello", "thanks", "thank you", "ok", "okay", "cool",
    "who are you", "what can you do", "help", "what is this",
)


def is_trivial(message: str) -> bool:
    """True for greetings and meta questions that should skip retrieval."""
    normalized = message.strip().lower().rstrip("?!.")
    if len(normalized) <= 3:
        return True
    return any(normalized == p or normalized.startswith(p + " ") for p in TRIVIAL_PATTERNS)
