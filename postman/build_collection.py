"""Build postman/lenny.postman_collection.json from readable Python.

Hand-editing a Postman export means editing JavaScript inside JSON strings.
The tests live here instead, and the JSON is generated and committed, so the
collection still imports into the Postman app as-is. CI runs this with
--check and fails if the committed JSON is out of date.

    python postman/build_collection.py           # regenerate
    python postman/build_collection.py --check   # verify, for CI

The run is ordered: it creates a session, chats in it, reads back what was
saved, and deletes it -- each step passing ids to the next through
collection variables. Run it against a fresh database seeded with
`python -m devtools.seed_fixture` and the deterministic mock LLM
(`python -m devtools.mock_llm`); see postman/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).with_name("lenny.postman_collection.json")
MISSING = "00000000-0000-0000-0000-000000000000"

# --------------------------------------------------------------------------
# JSON schemas for response bodies. `additionalProperties` is left open on
# purpose: the contract is "these fields exist with these types", and adding
# a field is not a breaking change for a client.
# --------------------------------------------------------------------------

UUID = {"type": "string", "pattern": "^[0-9a-f-]{36}$"}
NULLABLE_STR = {"type": ["string", "null"]}

ERROR = {
    "type": "object", "required": ["error"],
    "properties": {"error": {
        "type": "object", "required": ["code", "message", "hint", "request_id"],
        "properties": {"code": {"type": "string"}, "message": {"type": "string"},
                       "hint": {"type": "string"}, "request_id": {"type": "string"}},
    }},
}
SESSION_SUMMARY = {
    "type": "object",
    "required": ["id", "title", "created_at", "updated_at", "message_count"],
    "properties": {"id": UUID, "title": {"type": "string"}, "created_at": {"type": "string"},
                   "updated_at": {"type": "string"}, "message_count": {"type": "integer"}},
}
CITATION = {
    "type": "object", "required": ["n", "chunk_id", "episode_title", "guest", "url"],
    "properties": {"n": {"type": ["integer", "null"]}, "chunk_id": {"type": "string"},
                   "episode_title": {"type": "string"}, "guest": {"type": "string"},
                   "url": {"type": "string"}, "timestamp": {"type": "string"},
                   "publish_date": NULLABLE_STR, "similarity": {"type": "number"}},
}
MESSAGE = {
    "type": "object", "required": ["id", "role", "content", "citations", "created_at"],
    "properties": {"id": UUID, "role": {"enum": ["user", "assistant", "tool"]},
                   "content": {"type": "string"},
                   "citations": {"type": "array", "items": CITATION},
                   "provider": NULLABLE_STR, "model": NULLABLE_STR,
                   "latency_ms": {"type": ["integer", "null"]}},
}
ARTIFACT_SUMMARY = {
    "type": "object", "required": ["id", "kind", "title", "created_at"],
    "properties": {"id": UUID, "kind": {"enum": ["markdown", "html"]},
                   "title": {"type": "string"}, "created_at": {"type": "string"}},
}
ARTIFACT = {**ARTIFACT_SUMMARY,
            "required": ARTIFACT_SUMMARY["required"] + ["content"],
            "properties": {**ARTIFACT_SUMMARY["properties"], "content": {"type": "string"},
                           "sanitizer_report": {"type": ["object", "null"]}}}
SESSION_DETAIL = {
    **SESSION_SUMMARY,
    "required": SESSION_SUMMARY["required"] + ["messages", "artifacts"],
    "properties": {**SESSION_SUMMARY["properties"],
                   "messages": {"type": "array", "items": MESSAGE},
                   "artifacts": {"type": "array", "items": ARTIFACT_SUMMARY}},
}
CHAT = {
    "type": "object",
    "required": ["session_id", "message", "artifacts", "tool_calls", "grounded",
                 "provider", "model", "latency_ms"],
    "properties": {
        "session_id": UUID, "message": MESSAGE,
        "artifacts": {"type": "array", "items": ARTIFACT},
        "tool_calls": {"type": "array", "items": {
            "type": "object", "required": ["tool", "ok", "latency_ms"],
            "properties": {"tool": {"type": "string"}, "ok": {"type": "boolean"},
                           "latency_ms": {"type": "integer"}}}},
        "grounded": {"type": "boolean"}, "provider": {"type": "string"},
        "model": {"type": "string"}, "latency_ms": {"type": "integer"},
    },
}
SEARCH = {
    "type": "object", "required": ["query", "grounded", "best_similarity", "latency_ms", "results"],
    "properties": {
        "query": {"type": "string"}, "grounded": {"type": "boolean"},
        "best_similarity": {"type": "number"}, "latency_ms": {"type": "integer"},
        "results": {"type": "array", "items": {
            "type": "object",
            "required": ["chunk_id", "guest", "episode_title", "excerpt", "similarity"],
            "properties": {"guest": {"type": "string"}, "excerpt": {"type": "string"},
                           "similarity": {"type": "number"}}}},
    },
}
HEALTH = {"type": "object", "required": ["status"], "properties": {"status": {"const": "ok"}}}
DEEP_HEALTH = {
    "type": "object", "required": ["status", "database", "providers", "knowledge_base", "config"],
    "properties": {
        "status": {"enum": ["ok", "degraded", "error"]},
        "database": {"type": "object", "required": ["healthy"]},
        "providers": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["provider", "model", "healthy"]}},
        "knowledge_base": {"type": "object",
                           "required": ["episodes", "chunks", "embedded_chunks", "ready"]},
    },
}
CONFIG = {
    "type": "object",
    "required": ["provider", "model", "endpoint", "is_local", "knowledge_base_ready",
                 "episodes", "chunks"],
    "properties": {"provider": {"type": "string"}, "model": {"type": "string"},
                   "is_local": {"type": "boolean"}, "knowledge_base_ready": {"type": "boolean"},
                   "episodes": {"type": "integer"}, "chunks": {"type": "integer"}},
}
TRACE = {"type": "array", "items": {
    "type": "object", "required": ["id", "session_id", "request_id", "kind", "name", "duration_ms"],
    "properties": {"kind": {"enum": ["llm_call", "tool_call"]}, "duration_ms": {"type": "integer"}}}}

# --------------------------------------------------------------------------
# test-script fragments
# --------------------------------------------------------------------------


def status(code: int) -> str:
    return f'pm.test("status is {code}", () => pm.response.to.have.status({code}));'


def schema(s: dict, name: str = "body") -> str:
    return (f"pm.test({json.dumps(name + ' matches its JSON schema')}, () => "
            f"pm.response.to.have.jsonSchema({json.dumps(s)}));")


def fast(var: str = "maxResponseMs") -> str:
    return (f'pm.test("responds within " + pm.environment.get("{var}") + " ms", () => '
            f'pm.expect(pm.response.responseTime).to.be.below(Number(pm.environment.get("{var}"))));')


def error(code: int, error_code: str) -> list[str]:
    return [status(code), schema(ERROR, "error envelope"),
            f'pm.test("error code is {error_code}", () => '
            f'pm.expect(pm.response.json().error.code).to.eql("{error_code}"));', fast()]


def js(*lines: str) -> list[str]:
    return [line for block in lines for line in block.split("\n")]


def request(name: str, method: str, path: str, tests: list[str], body: dict | None = None,
            headers: dict | None = None) -> dict:
    url_raw = "{{baseUrl}}" + path
    host_path, _, query = path.partition("?")
    url: dict = {"raw": url_raw, "host": ["{{baseUrl}}"],
                 "path": [p for p in host_path.strip("/").split("/") if p]}
    if query:
        url["query"] = [{"key": k, "value": v} for k, _, v in
                        (pair.partition("=") for pair in query.split("&"))]
    req: dict = {"method": method, "header": [], "url": url}
    hdrs = dict(headers or {})
    if body is not None:
        hdrs.setdefault("Content-Type", "application/json")
        req["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2),
                       "options": {"raw": {"language": "json"}}}
    req["header"] = [{"key": k, "value": v} for k, v in hdrs.items()]
    return {"name": name, "request": req,
            "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": js(*tests)}}]}


def folder(name: str, description: str, items: list[dict]) -> dict:
    return {"name": name, "description": description, "item": items}


# --------------------------------------------------------------------------
# the collection
# --------------------------------------------------------------------------

S = "/api/sessions/{{sessionId}}"

ITEMS = [
    folder("Health and config", "Liveness, dependency status and UI config.", [
        request("Liveness", "GET", "/health", [status(200), schema(HEALTH), fast()]),
        request("Deep health: database, provider and knowledge base", "GET", "/health/deep", [
            status(200), schema(DEEP_HEALTH), fast(),
            'pm.test("database is healthy", () => pm.expect(pm.response.json().database.healthy).to.be.true);',
            'pm.test("the seeded knowledge base is ready", () => pm.expect(pm.response.json().knowledge_base.ready).to.be.true);',
        ]),
        request("UI config", "GET", "/api/config", [
            status(200), schema(CONFIG), fast(),
            'pm.test("reports the seeded corpus", () => pm.expect(pm.response.json().episodes).to.be.above(0));',
        ]),
        request("Unknown route is a typed 404", "GET", "/api/does-not-exist", error(404, "http_error")),
    ]),
    folder("Sessions", "Create a session and keep its id for the rest of the run.", [
        request("Create session", "POST", "/api/sessions", [
            status(201), schema(SESSION_SUMMARY), fast(),
            'pm.test("starts empty", () => pm.expect(pm.response.json().message_count).to.eql(0));',
            'pm.collectionVariables.set("sessionId", pm.response.json().id);',
        ], body={"title": "Newman run"}),
        request("List sessions includes it", "GET", "/api/sessions?limit=50", [
            status(200), fast(),
            schema({"type": "array", "items": SESSION_SUMMARY}),
            'pm.test("contains the new session", () => pm.expect(pm.response.json().map(s => s.id)).to.include(pm.collectionVariables.get("sessionId")));',
        ]),
        request("Get session", "GET", S, [
            status(200), schema(SESSION_DETAIL), fast(),
            'pm.test("has the requested title", () => pm.expect(pm.response.json().title).to.eql("Newman run"));',
        ]),
        request("Missing session is 404", "GET", f"/api/sessions/{MISSING}", error(404, "session_not_found")),
        request("Malformed session id is 422", "GET", "/api/sessions/not-a-uuid", error(422, "validation_error")),
        request("Title over 200 characters is 422", "POST", "/api/sessions", error(422, "validation_error"),
                body={"title": "x" * 201}),
        request("limit=0 is 422", "GET", "/api/sessions?limit=0", error(422, "validation_error")),
    ]),
    folder("Search", "Retrieval only, no model: hybrid search, the relevance floor and filters.", [
        request("Relevant query is grounded", "POST", "/api/search", [
            status(200), schema(SEARCH), fast(),
            'pm.test("clears the relevance floor", () => pm.expect(pm.response.json().grounded).to.be.true);',
        ], body={"query": "How do I improve user retention through onboarding?", "top_k": 3}),
        request("Guest filter returns only that guest", "POST", "/api/search", [
            status(200), schema(SEARCH), fast(),
            'pm.test("every result is Casey Winters", () => pm.response.json().results.forEach(r => pm.expect(r.guest).to.eql("Casey Winters")));',
        ], body={"query": "retention and growth loops", "guest": "casey"}),
        request("Date range excludes other years", "POST", "/api/search", [
            status(200), schema(SEARCH), fast(),
            'pm.test("only 2024 episodes", () => pm.response.json().results.forEach(r => pm.expect(r.publish_date).to.match(/^2024-/)));',
        ], body={"query": "acquisition channels", "since": "2024-01-01", "until": "2024-12-31"}),
        request("Off-topic query is not grounded", "POST", "/api/search", [
            status(200), schema(SEARCH), fast(),
            'pm.test("refuses to ground", () => pm.expect(pm.response.json().grounded).to.be.false);',
        ], body={"query": "best sourdough starter recipe"}),
        request("top_k=0 is 422", "POST", "/api/search", error(422, "validation_error"),
                body={"query": "retention", "top_k": 0}),
        request("since after until is 422", "POST", "/api/search", error(422, "validation_error"),
                body={"query": "retention", "since": "2024-01-01", "until": "2023-01-01"}),
        request("Empty query is 422", "POST", "/api/search", error(422, "validation_error"),
                body={"query": ""}),
    ]),
    folder("Chat", "A full agent turn each. The model is the deterministic mock.", [
        request("Grounded answer cites what it used", "POST", f"{S}/chat", [
            status(200), schema(CHAT), fast("maxChatMs"),
            'const r = pm.response.json();',
            'pm.test("is grounded", () => pm.expect(r.grounded).to.be.true);',
            'pm.test("searched first", () => pm.expect(r.tool_calls.map(t => t.tool)).to.include("search_transcripts"));',
            'pm.test("every [n] in the text has a citation numbered n", () => {',
            '  const cited = [...r.message.content.matchAll(/\\[(\\d+)\\]/g)].map(m => Number(m[1]));',
            '  pm.expect(cited.length).to.be.above(0);',
            '  pm.expect(r.message.citations.map(c => c.n)).to.have.members([...new Set(cited)]);',
            '});',
        ], body={"message": "How do I improve user retention through onboarding?"}),
        request("Off-topic question is refused with no sources", "POST", f"{S}/chat", [
            status(200), schema(CHAT), fast("maxChatMs"),
            'const r = pm.response.json();',
            'pm.test("not grounded", () => pm.expect(r.grounded).to.be.false);',
            'pm.test("carries no sources", () => pm.expect(r.message.citations).to.be.empty);',
        ], body={"message": "What is the best sourdough starter recipe?"}),
        request("Document request creates an artifact", "POST", f"{S}/chat", [
            status(200), schema(CHAT), fast("maxChatMs"),
            'const r = pm.response.json();',
            'pm.test("one markdown artifact", () => { pm.expect(r.artifacts).to.have.length(1); pm.expect(r.artifacts[0].kind).to.eql("markdown"); });',
            'pm.collectionVariables.set("artifactId", r.artifacts[0].id);',
        ], body={"message": "Write a checklist document on how onboarding improves user retention"}),
        request("Blank message is 422", "POST", f"{S}/chat", error(422, "validation_error"),
                body={"message": "   "}),
        request("Chat on a missing session is 404", "POST", f"/api/sessions/{MISSING}/chat",
                error(404, "session_not_found"), body={"message": "hi"}),
    ]),
    folder("Chat stream", "The same turn over Server-Sent Events.", [
        request("Streams progress, then the saved turn", "POST", f"{S}/chat/stream", [
            status(200), fast("maxChatMs"),
            'pm.test("is an event stream", () => pm.expect(pm.response.headers.get("Content-Type")).to.include("text/event-stream"));',
            'const events = pm.response.text().split("\\n\\n").filter(b => b.startsWith("event: ")).map(b => {',
            '  const lines = b.split("\\n");',
            '  return JSON.parse(lines.find(l => l.startsWith("data: ")).slice(6));',
            '});',
            'pm.test("reports the search before answering", () => pm.expect(events.map(e => e.type)).to.include.members(["thinking", "tool_start", "tool_end"]));',
            'pm.test("ends with done", () => pm.expect(events[events.length - 1].type).to.eql("done"));',
            f'pm.test("done carries a chat response", () => pm.expect(events[events.length - 1].response).to.be.jsonSchema({json.dumps(CHAT)}));',
        ], body={"message": "Why does retention matter for growth loops?"}),
        request("Validation fails before the stream opens", "POST", f"{S}/chat/stream",
                error(422, "validation_error"), body={}),
    ]),
    folder("Session history", "What the chats above saved.", [
        request("Session holds every turn", "GET", S, [
            status(200), schema(SESSION_DETAIL), fast(),
            'const s = pm.response.json();',
            'pm.test("4 user turns and 4 replies", () => pm.expect(s.message_count).to.eql(8));',
            'pm.test("titled by its first question", () => pm.expect(s.title).to.include("retention"));',
            'pm.test("lists the artifact", () => pm.expect(s.artifacts.map(a => a.id)).to.include(pm.collectionVariables.get("artifactId")));',
        ]),
    ]),
    folder("Artifacts", "Documents created by the chat above.", [
        request("List by session", "GET", "/api/artifacts?session_id={{sessionId}}", [
            status(200), fast(), schema({"type": "array", "items": ARTIFACT_SUMMARY}),
            'pm.test("contains the created artifact", () => pm.expect(pm.response.json().map(a => a.id)).to.include(pm.collectionVariables.get("artifactId")));',
        ]),
        request("Get artifact content", "GET", "/api/artifacts/{{artifactId}}", [
            status(200), schema(ARTIFACT), fast(),
            'pm.test("has rendered content", () => pm.expect(pm.response.json().content).to.include("# "));',
        ]),
        request("Missing artifact is 404", "GET", f"/api/artifacts/{MISSING}", error(404, "artifact_not_found")),
        request("Malformed artifact id is 422", "GET", "/api/artifacts/not-a-uuid", error(422, "validation_error")),
    ]),
    folder("Traces", "Every model and tool call the chats above made.", [
        request("Session trace", "GET", f"{S}/trace", [
            status(200), schema(TRACE), fast(),
            'pm.test("recorded both model and tool calls", () => {',
            '  const kinds = pm.response.json().map(s => s.kind);',
            '  pm.expect(kinds).to.include("llm_call"); pm.expect(kinds).to.include("tool_call");',
            '});',
        ]),
        request("limit over 1000 is 422", "GET", f"{S}/trace?limit=1001", error(422, "validation_error")),
    ]),
    folder("Cleanup", "Delete the session and confirm it is gone.", [
        request("Delete session", "DELETE", S, [
            status(204), fast(),
            'pm.test("no body", () => pm.expect(pm.response.text()).to.eql(""));',
        ]),
        request("Deleted session is 404", "GET", S, error(404, "session_not_found")),
        request("Its artifact went with it", "GET", "/api/artifacts/{{artifactId}}",
                error(404, "artifact_not_found")),
        request("Deleting again is 404", "DELETE", S, error(404, "session_not_found")),
    ]),
]

COLLECTION = {
    "info": {
        "name": "Lenny Growth Assistant API",
        "description": "Every route in backend/app/api, in run order. Generated by "
                       "postman/build_collection.py -- edit that, not this file.",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
    },
    "item": ITEMS,
    "variable": [{"key": "sessionId", "value": ""}, {"key": "artifactId", "value": ""}],
}


def render() -> str:
    return json.dumps(COLLECTION, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.name} is out of date: run python postman/build_collection.py")
            return 1
        print(f"{OUT.name} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8")
    requests = sum(len(f["item"]) for f in ITEMS)
    print(f"wrote {OUT.name}: {len(ITEMS)} folders, {requests} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
