"""A deterministic stand-in for an OpenAI-compatible model, for API and E2E runs.

The app talks to it through its ordinary `openai_compat` provider
(LLM_PROVIDER=openai_compat, LLM_BASE_URL=http://127.0.0.1:<port>/v1), so
nothing in the application has a test mode: the real agent loop, tools,
guards and persistence all run. Only the model's judgement is scripted:

  no tools offered (a trivial message)   -> a short greeting
  nothing searched yet this turn          -> call search_transcripts(<question>)
  search found nothing                    -> refuse, citing nothing
  question asks for a document            -> call create_artifact, citing [n]
  document already created                -> a one-line description
  otherwise                               -> answer from the first excerpt, citing [n]

Standard library only, so it runs anywhere Python does:

    python -m devtools.mock_llm --port 9999
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model"
DOCUMENT_WORDS = ("checklist", "document", "one-pager", "one pager", "template")


def respond(payload: dict) -> dict:
    """The assistant message for one chat-completions request."""
    messages = payload.get("messages") or []
    tools_offered = bool(payload.get("tools"))
    tool_results = [m for m in messages if m.get("role") == "tool"]

    # Tool results only ever belong to the current turn (the app replays
    # prior turns as user/assistant text), so the question is the last user
    # message before the first tool result -- later user messages are the
    # app's own guard nudges.
    first_tool = next((i for i, m in enumerate(messages) if m.get("role") == "tool"), len(messages))
    question = next((m.get("content") or "" for m in reversed(messages[:first_tool])
                     if m.get("role") == "user"), "")

    if not tools_offered:
        return _text("Hi! Ask me anything about product, growth or retention.")

    searches = [_json(m) for m in tool_results if m.get("name") == "search_transcripts"]
    if not searches:
        return _call("search_transcripts", {"query": question[:200]})

    latest = searches[-1]
    hits = latest.get("results") or []
    if not latest.get("grounded") or not hits:
        return _text("Lenny's Podcast transcripts don't cover that, so I can't answer it "
                     "from the corpus.")

    first = hits[0]
    wants_document = any(w in question.lower() for w in DOCUMENT_WORDS)
    made_document = any(m.get("name") == "create_artifact" for m in tool_results)
    if wants_document and not made_document:
        return _call("create_artifact", {
            "kind": "markdown",
            "title": "Onboarding checklist",
            "content": f"# Onboarding checklist\n\n- Start with the first session "
                       f"every new user sees [{first['n']}]\n- Measure retention by cohort\n",
        })
    if made_document:
        return _text("I put a one-page onboarding checklist in the panel beside the chat.")

    return _text(f"{first['guest']} argues that onboarding is the lever for retention "
                 f"[{first['n']}].")


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _call(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"call_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}


def _json(message: dict) -> dict:
    try:
        return json.loads(message.get("content") or "{}")
    except json.JSONDecodeError:
        return {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        message = respond(payload)
        self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # quiet: CI logs are for the app
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    print(f"mock LLM on http://{args.host}:{args.port}/v1", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
