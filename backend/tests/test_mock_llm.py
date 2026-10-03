"""The mock LLM that API and E2E runs use in place of a real model.

Those suites are only deterministic if this is, so each branch of its script
is pinned here -- including that it reads the question from the user's own
message rather than from a guard nudge the app appended later.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from devtools.mock_llm import Handler, respond

pytestmark = pytest.mark.unit

TOOLS = [{"type": "function", "function": {"name": "search_transcripts"}}]


def search_result(grounded: bool) -> dict:
    results = [{"n": 3, "guest": "Adam Fishman"}] if grounded else []
    return {"role": "tool", "name": "search_transcripts",
            "content": json.dumps({"grounded": grounded, "results": results})}


def ask(question: str, *after: dict, tools=TOOLS) -> dict:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": question}, *after]
    return respond({"messages": messages, "tools": tools})


def called(message: dict) -> tuple[str, dict]:
    fn = message["tool_calls"][0]["function"]
    return fn["name"], json.loads(fn["arguments"])


def test_no_tools_means_a_greeting():
    assert "Hi!" in ask("hi", tools=None)["content"]


def test_searches_first_with_the_question():
    assert called(ask("How do I improve retention?")) == (
        "search_transcripts", {"query": "How do I improve retention?"})


def test_answers_citing_the_first_excerpt_by_its_number():
    reply = ask("How do I improve retention?", search_result(True))
    assert reply["content"] == "Adam Fishman argues that onboarding is the lever for retention [3]."


def test_refuses_when_nothing_was_found():
    reply = ask("Sourdough?", search_result(False))
    assert "don't cover that" in reply["content"]
    assert "[" not in reply["content"]


def test_document_request_creates_an_artifact_citing_its_source():
    name, args = called(ask("Write a checklist on onboarding", search_result(True)))
    assert name == "create_artifact"
    assert "[3]" in args["content"]


def test_then_describes_the_document_instead_of_repeating_it():
    created = {"role": "tool", "name": "create_artifact", "content": "{}"}
    reply = ask("Write a checklist on onboarding", search_result(True), created,
                {"role": "user", "content": "Reply with one sentence."})
    assert "panel" in reply["content"]


def test_a_guard_nudge_is_not_mistaken_for_the_question():
    """The app appends user-role nudges after tool results; the document
    decision must still come from what the user asked."""
    reply = ask("How do I improve retention?", search_result(True),
                {"role": "user", "content": "Put this in a document checklist."})
    assert "tool_calls" not in reply


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_speaks_the_openai_chat_completions_shape(server):
    body = httpx.post(f"{server}/chat/completions",
                      json={"messages": [{"role": "user", "content": "q"}], "tools": TOOLS}).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "search_transcripts"


def test_models_endpoint_answers_health_probes(server):
    assert httpx.get(f"{server}/models").json()["data"][0]["id"] == "mock-model"


def test_unknown_paths_are_404(server):
    assert httpx.get(f"{server}/nope").status_code == 404
