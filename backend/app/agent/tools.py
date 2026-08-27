"""Tool registry — the agent's skill boundaries.

Three tools, deliberately few. Every additional tool measurably degrades routing
accuracy on small local models, which are the required demo target, so each one
here has to earn its place:

  search_transcripts  — the grounding primitive. Everything factual goes through it.
  write_ship30_essay  — long-form generation with its own pipeline and quality gate.
  create_artifact     — turns the conversation into a rendered document.

Each tool returns a JSON-serialisable dict that goes back to the model verbatim,
plus side-effects recorded on the ToolContext for the API layer to persist.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.base import ToolSpec
from app.logging import get_logger
from app.rag.retriever import search
from app.security.sanitize import sanitize_artifact
from app.skills.ship30.skill import write_ship30_essay

log = get_logger(__name__)

MAX_ARTIFACT_BYTES = 200_000


@dataclass
class PendingArtifact:
    id: str
    kind: str
    title: str
    content: str
    sanitizer_report: dict


@dataclass
class ToolContext:
    """Per-turn state threaded through tool calls."""

    db: AsyncSession
    session_id: uuid.UUID
    citations: list[dict] = field(default_factory=list)
    artifacts: list[PendingArtifact] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    grounded: bool = False
    searched: bool = False

    def add_citations(self, new: list[dict]) -> None:
        """Merge, preserving order and de-duplicating by chunk."""
        seen = {c["chunk_id"] for c in self.citations}
        for c in new:
            if c["chunk_id"] not in seen:
                self.citations.append(c)
                seen.add(c["chunk_id"])


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="search_transcripts",
        description=(
            "Search Lenny's Podcast transcripts for evidence. Call this before answering "
            "ANY question about product management, growth, retention, pricing, hiring, "
            "strategy, or company building. Returns numbered excerpts with sources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Rewrite the user's question into a "
                        "self-contained search phrase, resolving pronouns from the "
                        "conversation (e.g. 'how does he suggest hiring PMs' -> "
                        "'hiring product managers first PM hire')."
                    ),
                }
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="write_ship30_essay",
        description=(
            "Write a ~1,250-word Ship 30 for 30-style essay grounded in the transcripts. "
            "Use this ONLY when the user explicitly asks for an essay, article, blog post, "
            "or long-form written piece. Do not use it for ordinary questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The essay topic, as a specific and self-contained phrase.",
                }
            },
            "required": ["topic"],
        },
    ),
    ToolSpec(
        name="create_artifact",
        description=(
            "Create a rendered document shown beside the chat. Use when the user asks for "
            "a document, report, one-pager, table, checklist, template, or an HTML/CSS "
            "page. Pass the COMPLETE final content, not a description of it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["markdown", "html"],
                    "description": (
                        "'markdown' for documents and reports; 'html' only when the user "
                        "wants styling, layout, or a web page."
                    ),
                },
                "title": {"type": "string", "description": "Short title for the document."},
                "content": {
                    "type": "string",
                    "description": (
                        "The complete document. For html, a full self-contained snippet "
                        "with a <style> block. External scripts and network requests are "
                        "stripped by the renderer, so inline everything."
                    ),
                },
            },
            "required": ["kind", "title", "content"],
        },
    ),
]

TOOL_SPECS_BY_NAME = {t.name: t for t in TOOL_SPECS}


async def _tool_search_transcripts(ctx: ToolContext, args: dict[str, Any]) -> dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "query is required", "results": []}

    result = await search(ctx.db, query)
    ctx.searched = True

    if not result.grounded:
        # This is the grounding guard doing its job. We tell the model plainly
        # that it has nothing, so the honest answer is the only available one.
        return {
            "results": [],
            "grounded": False,
            "instruction": (
                "No transcript passages met the relevance threshold for this query. "
                "Tell the user directly that Lenny's Podcast transcripts do not cover "
                "this topic. Do NOT answer from your own knowledge. You may suggest a "
                "related topic the corpus does cover."
            ),
        }

    ctx.grounded = True
    ctx.add_citations([c.as_citation() for c in result.chunks])

    return {
        "grounded": True,
        "results": [
            {
                "n": i,
                "guest": c.guest,
                "episode": c.title,
                "timestamp": c.timestamp_label,
                "excerpt": c.text,
            }
            for i, c in enumerate(result.chunks, start=1)
        ],
        "instruction": (
            "Answer using only these excerpts. Cite them inline as [1], [2] matching "
            "the 'n' field. Attribute ideas to the guest who said them."
        ),
    }


async def _tool_write_ship30_essay(ctx: ToolContext, args: dict[str, Any]) -> dict:
    topic = (args.get("topic") or "").strip()
    if not topic:
        return {"error": "topic is required"}

    result = await write_ship30_essay(ctx.db, topic)
    if not result["ok"]:
        return {"ok": False, "message": result["message"]}

    ctx.grounded = True
    ctx.add_citations(result["citations"])

    # The essay is long. Rather than push 1,250 words back through the model --
    # which on a 3B local model risks it truncating or "summarising" the work --
    # we register it as an artifact and hand the model only a short receipt.
    artifact = PendingArtifact(
        id=str(uuid.uuid4()),
        kind="markdown",
        title=_title_from_markdown(result["essay"], fallback=topic),
        content=result["essay"],
        sanitizer_report={"sanitizer": "none", "findings": [], "modified": False},
    )
    ctx.artifacts.append(artifact)

    return {
        "ok": True,
        "artifact_created": True,
        "title": artifact.title,
        "word_count": result["rubric"]["word_count"],
        "rubric_passed": result["rubric"]["passed"],
        "instruction": (
            "The essay has been written and is now displayed in the artifact panel "
            "beside the chat. Reply with two or three sentences describing what the "
            "essay argues and which guests it draws on. Do NOT reproduce the essay."
        ),
    }


async def _tool_create_artifact(ctx: ToolContext, args: dict[str, Any]) -> dict:
    kind = (args.get("kind") or "markdown").strip().lower()
    title = (args.get("title") or "Untitled").strip()[:300]
    content = args.get("content") or ""

    if kind not in ("markdown", "html"):
        return {"error": f"kind must be 'markdown' or 'html', got '{kind}'"}
    if not content.strip():
        return {"error": "content is empty. Pass the complete document text."}
    if len(content.encode("utf-8")) > MAX_ARTIFACT_BYTES:
        return {"error": f"content exceeds the {MAX_ARTIFACT_BYTES} byte limit."}

    safe, report = sanitize_artifact(kind, content)
    artifact = PendingArtifact(
        id=str(uuid.uuid4()), kind=kind, title=title, content=safe, sanitizer_report=report
    )
    ctx.artifacts.append(artifact)

    if report["findings"]:
        log.warning(
            "artifact_sanitized",
            session_id=str(ctx.session_id),
            kind=kind,
            findings=report["findings"],
        )

    return {
        "ok": True,
        "artifact_created": True,
        "title": title,
        "kind": kind,
        "sanitizer_findings": report["findings"],
        "instruction": (
            "The document is now rendered in the artifact panel beside the chat. "
            "Reply with one or two sentences about what you made. Do NOT repeat "
            "its contents in the chat."
        ),
    }


def _title_from_markdown(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:300]
    return fallback[:300]


TOOL_IMPLEMENTATIONS = {
    "search_transcripts": _tool_search_transcripts,
    "write_ship30_essay": _tool_write_ship30_essay,
    "create_artifact": _tool_create_artifact,
}


async def execute_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict:
    """Dispatch one tool call. Never raises -- errors are returned to the model."""
    started = time.perf_counter()

    impl = TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        # Recorded, not just returned: a model hallucinating tool names is a
        # routing signal we want visible in the turn's trace, not swallowed.
        log.warning("unknown_tool_called", tool=name)
        ctx.tool_log.append({"tool": name, "args": args, "ok": False, "latency_ms": 0})
        return {
            "error": f"Unknown tool '{name}'. Available: {', '.join(TOOL_IMPLEMENTATIONS)}."
        }

    try:
        result = await impl(ctx, args)
        ok = "error" not in result
    except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the turn
        log.exception("tool_execution_failed", tool=name)
        result, ok = {"error": f"{type(exc).__name__}: {exc}"}, False

    elapsed = int((time.perf_counter() - started) * 1000)
    ctx.tool_log.append({"tool": name, "args": args, "ok": ok, "latency_ms": elapsed})
    log.info("tool_executed", tool=name, ok=ok, latency_ms=elapsed)
    return result
