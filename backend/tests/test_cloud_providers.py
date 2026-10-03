"""The Bedrock and Azure OpenAI adapters, against mocked HTTP.

Every request goes to an `httpx.MockTransport`, so nothing here needs an
account or reaches the network. What is checked is the part that is ours: the
request each adapter builds, the translation of messages and tool calls, the
response parsing, and the mapping of failures onto the app's typed errors.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from app.config import get_settings
from app.llm.azure_openai import AzureOpenAIProvider
from app.llm.base import (
    ChatMessage,
    LLMAuthError,
    LLMBadResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    ToolCall,
    ToolSpec,
)
from app.llm.bedrock import BedrockProvider, to_converse
from app.llm.registry import build_provider
from app.llm.sigv4 import SigV4Auth

pytestmark = pytest.mark.unit

SEARCH = ToolSpec("search_transcripts", "Search the corpus.",
                  {"type": "object", "properties": {"query": {"type": "string"}}})

# A tool loop as the agent builds it: system prompt, question, a turn that
# calls two tools, their two results, then a nudge from a guard.
LOOP = [
    ChatMessage("system", "You are a growth advisor."),
    ChatMessage("user", "How do I improve retention?"),
    ChatMessage("assistant", "", tool_calls=[
        ToolCall("t1", "search_transcripts", {"query": "retention"}),
        ToolCall("t2", "search_transcripts", {"query": "onboarding"}),
    ]),
    ChatMessage("tool", '{"results": []}', tool_call_id="t1", name="search_transcripts"),
    ChatMessage("tool", '{"results": [1]}', tool_call_id="t2", name="search_transcripts"),
    ChatMessage("user", "Cite your sources."),
]


class Recorder:
    """A mock transport that records each request and replies as scripted."""

    def __init__(self, reply: httpx.Response | Exception) -> None:
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


def configure(monkeypatch, **env: str) -> None:
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_SESSION_TOKEN", "AZURE_OPENAI_API_KEY"):
        monkeypatch.setenv(name, "")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


# ------------------------------------------------------------------ SigV4

def test_sigv4_matches_the_worked_example_in_the_aws_documentation():
    """The IAM ListUsers example from AWS's SigV4 guide, with its published
    signature. If this passes, the canonical request, string to sign and
    signing key are all built the way AWS builds them."""
    auth = SigV4Auth("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "us-east-1", "iam")
    request = httpx.Request(
        "GET", "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                 "X-Amz-Date": "20150830T123600Z"})
    auth.sign(request)
    assert request.headers["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, "
        "SignedHeaders=content-type;host;x-amz-date, "
        "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7")


def test_sigv4_signs_a_session_token():
    auth = SigV4Auth("AKID", "secret", "us-east-1", "bedrock", session_token="tok",
                     clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    request = httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/x",
                            content=b"{}")
    auth.sign(request)
    assert request.headers["x-amz-security-token"] == "tok"
    assert request.headers["x-amz-date"] == "20260102T030405Z"
    assert "SignedHeaders=host;x-amz-date;x-amz-security-token," in request.headers["authorization"]


def test_a_changed_body_changes_the_signature():
    def signature(body: bytes) -> str:
        auth = SigV4Auth("AKID", "secret", "us-east-1", "bedrock",
                         clock=lambda: datetime(2026, 1, 2, tzinfo=UTC))
        request = httpx.Request("POST", "https://example.com/", content=body)
        auth.sign(request)
        return request.headers["authorization"].rsplit("=", 1)[1]

    assert signature(b'{"a": 1}') != signature(b'{"a": 2}')


# ------------------------------------------------- Bedrock: translation

def test_converse_body_for_a_tool_loop():
    body = to_converse(LOOP, [SEARCH])

    assert body["system"] == [{"text": "You are a growth advisor."}]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    # The assistant turn: its empty text is dropped, its calls become toolUse.
    assert body["messages"][1]["content"] == [
        {"toolUse": {"toolUseId": "t1", "name": "search_transcripts",
                     "input": {"query": "retention"}}},
        {"toolUse": {"toolUseId": "t2", "name": "search_transcripts",
                     "input": {"query": "onboarding"}}},
    ]
    # Both results and the nudge merge into one user turn, since roles alternate.
    results = body["messages"][2]["content"]
    assert [b["toolResult"]["toolUseId"] for b in results[:2]] == ["t1", "t2"]
    assert results[2] == {"text": "Cite your sources."}
    assert body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == SEARCH.parameters
    assert body["toolConfig"]["toolChoice"] == {"auto": {}}


def test_converse_body_without_tools_sends_earlier_tool_use_as_text():
    """The agent's closing call offers no tools, and Converse rejects tool
    blocks without a tool config -- so the history goes as plain text."""
    body = to_converse(LOOP, None)

    assert "toolConfig" not in body
    sent = json.dumps(body["messages"])
    assert "toolUse" not in sent and "toolResult" not in sent
    assert "(called search_transcripts with" in sent
    assert "Result of search_transcripts" in sent
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]


# ---------------------------------------------------- Bedrock: requests

CONVERSE_REPLY = {
    "output": {"message": {"role": "assistant", "content": [
        {"text": "Let me look that up."},
        {"toolUse": {"toolUseId": "tu-1", "name": "search_transcripts",
                     "input": {"query": "retention"}}},
    ]}},
    "stopReason": "tool_use",
    "usage": {"inputTokens": 120, "outputTokens": 18, "totalTokens": 138},
}


async def test_bedrock_chat_with_an_api_key(monkeypatch):
    configure(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="br-key", BEDROCK_REGION="eu-west-1",
              BEDROCK_MODEL_ID="amazon.nova-lite-v1:0")
    mock = Recorder(httpx.Response(200, json=CONVERSE_REPLY))
    provider = BedrockProvider(transport=mock.transport)

    response = await provider.chat(LOOP[:2], [SEARCH], temperature=0.2, max_tokens=500)

    request = mock.requests[0]
    assert str(request.url) == ("https://bedrock-runtime.eu-west-1.amazonaws.com"
                                "/model/amazon.nova-lite-v1%3A0/converse")
    assert request.headers["authorization"] == "Bearer br-key"
    assert mock.body["inferenceConfig"] == {"temperature": 0.2, "maxTokens": 500}
    assert response.content == "Let me look that up."
    assert response.tool_calls == [ToolCall("tu-1", "search_transcripts", {"query": "retention"})]
    assert (response.prompt_tokens, response.completion_tokens) == (120, 18)
    assert response.finish_reason == "tool_calls"
    assert (response.provider, response.model) == ("bedrock", "amazon.nova-lite-v1:0")


async def test_bedrock_chat_with_iam_credentials_is_signed(monkeypatch):
    configure(monkeypatch, AWS_ACCESS_KEY_ID="AKIDTEST", AWS_SECRET_ACCESS_KEY="secret",
              BEDROCK_REGION="us-east-1")
    mock = Recorder(httpx.Response(200, json={
        "output": {"message": {"content": [{"text": "Hello."}]}}, "stopReason": "end_turn"}))
    provider = BedrockProvider(transport=mock.transport)

    response = await provider.chat([ChatMessage("user", "hi")])

    auth = mock.requests[0].headers["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDTEST/")
    assert "/us-east-1/bedrock/aws4_request" in auth
    assert response.content == "Hello." and response.finish_reason == "stop"


async def test_bedrock_without_credentials_fails_before_any_request(monkeypatch):
    configure(monkeypatch)
    mock = Recorder(httpx.Response(200, json=CONVERSE_REPLY))
    provider = BedrockProvider(transport=mock.transport)

    with pytest.raises(LLMAuthError, match="AWS_BEARER_TOKEN_BEDROCK"):
        await provider.chat([ChatMessage("user", "hi")])
    assert mock.requests == []
    health = await provider.health()
    assert not health.healthy and health.detail == "no AWS credentials set"


@pytest.mark.parametrize("reply, error", [
    (httpx.Response(403, json={"message": "The security token included is invalid."}),
     LLMAuthError),
    (httpx.Response(429, json={"message": "Too many requests"}), LLMUnavailableError),
    (httpx.Response(503, json={"message": "Service unavailable"}), LLMUnavailableError),
    (httpx.Response(400, json={"message": "Malformed input request"}), LLMBadResponseError),
    (httpx.Response(200, text="not json"), LLMBadResponseError),
    (httpx.Response(200, json={"output": {}}), LLMBadResponseError),
    (httpx.ConnectError("refused"), LLMUnavailableError),
    (httpx.ReadTimeout("slow"), LLMTimeoutError),
])
async def test_bedrock_failures_map_to_typed_errors(monkeypatch, reply, error):
    """Typed, because the type decides fallback: unavailable and timeout fall
    back to another provider; auth and bad responses don't."""
    configure(monkeypatch, AWS_BEARER_TOKEN_BEDROCK="br-key")
    provider = BedrockProvider(transport=Recorder(reply).transport)
    with pytest.raises(error):
        await provider.chat([ChatMessage("user", "hi")])


# ------------------------------------------------------- Azure OpenAI

AZURE_ENV = {"AZURE_OPENAI_ENDPOINT": "https://lenny.openai.azure.com/",
             "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-mini-prod",
             "AZURE_OPENAI_API_KEY": "az-key"}


async def test_azure_chat_request_and_response(monkeypatch):
    configure(monkeypatch, **AZURE_ENV)
    mock = Recorder(httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_transcripts", "arguments": '{"query": "pricing"}'}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 9}}))
    provider = AzureOpenAIProvider(transport=mock.transport)

    response = await provider.chat(LOOP[:2], [SEARCH])

    request = mock.requests[0]
    assert str(request.url) == ("https://lenny.openai.azure.com/openai/deployments/"
                                "gpt-4o-mini-prod/chat/completions?api-version=2024-10-21")
    assert request.headers["api-key"] == "az-key"
    assert "authorization" not in request.headers
    assert mock.body["tools"][0]["function"]["name"] == "search_transcripts"
    assert response.tool_calls == [ToolCall("c1", "search_transcripts", {"query": "pricing"})]
    assert (response.provider, response.model) == ("azure_openai", "gpt-4o-mini-prod")


async def test_azure_without_a_key_names_the_right_variable(monkeypatch):
    configure(monkeypatch, **{**AZURE_ENV, "AZURE_OPENAI_API_KEY": ""})
    mock = Recorder(httpx.Response(200, json={}))
    provider = AzureOpenAIProvider(transport=mock.transport)

    with pytest.raises(LLMAuthError, match="AZURE_OPENAI_API_KEY is empty"):
        await provider.chat([ChatMessage("user", "hi")])
    assert mock.requests == []
    assert (await provider.health()).detail == "AZURE_OPENAI_API_KEY not set"


async def test_azure_health_probes_the_models_endpoint(monkeypatch):
    configure(monkeypatch, **AZURE_ENV)
    mock = Recorder(httpx.Response(200, json={"data": []}))
    health = await AzureOpenAIProvider(transport=mock.transport).health()

    assert health.healthy
    assert str(mock.requests[0].url) == ("https://lenny.openai.azure.com/openai/models"
                                         "?api-version=2024-10-21")


@pytest.mark.parametrize("status, error", [
    (401, LLMAuthError), (429, LLMUnavailableError), (400, LLMBadResponseError)])
async def test_azure_failures_map_to_typed_errors(monkeypatch, status, error):
    configure(monkeypatch, **AZURE_ENV)
    provider = AzureOpenAIProvider(transport=Recorder(httpx.Response(status, json={})).transport)
    with pytest.raises(error):
        await provider.chat([ChatMessage("user", "hi")])


# ------------------------------------------------------------ selection

@pytest.mark.parametrize("name, cls, model, endpoint", [
    ("bedrock", BedrockProvider, "amazon.nova-lite-v1:0",
     "https://bedrock-runtime.us-east-1.amazonaws.com"),
    ("azure_openai", AzureOpenAIProvider, "gpt-4o-mini-prod", "https://lenny.openai.azure.com/"),
])
async def test_llm_provider_selects_the_adapter(monkeypatch, name, cls, model, endpoint):
    configure(monkeypatch, LLM_PROVIDER=name, BEDROCK_REGION="us-east-1",
              BEDROCK_MODEL_ID="amazon.nova-lite-v1:0", AWS_BEARER_TOKEN_BEDROCK="br-key",
              **AZURE_ENV)
    provider = build_provider(name)
    try:
        assert isinstance(provider, cls) and provider.model == model
    finally:
        await provider.aclose()

    described = get_settings().describe_provider()
    assert described["provider"] == name
    assert described["model"] == model
    assert described["endpoint"] == endpoint
    assert described["api_key_present"] is True and described["is_local"] is False
