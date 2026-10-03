"""AWS Bedrock, through the Converse API.

Converse is Bedrock's one request shape for every chat model it hosts (Claude,
Nova, Llama, Mistral...), so one adapter covers them all, the way
`openai_compat` covers every OpenAI-style endpoint:

    POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/converse

What differs from the OpenAI shape, and is translated here:

- system prompts go in a top-level `system` list, not in `messages`;
- tool calls are `toolUse` content blocks, and tool results go back as
  `toolResult` blocks inside a *user* message;
- roles must alternate, so consecutive same-role messages (several tool results
  in a row) are merged into one;
- a conversation holding tool blocks must also declare the tools. When the
  agent makes its closing call without tools, earlier tool calls and results
  are sent as plain text instead.

Auth: a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) if set, otherwise a request
signed with IAM credentials from the environment (see `sigv4.py`).
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.llm.base import (
    ChatMessage,
    LLMAuthError,
    LLMBadResponseError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderHealth,
    ToolCall,
    ToolSpec,
)
from app.llm.sigv4 import SigV4Auth

# Converse's stopReason, in the OpenAI vocabulary the rest of the app logs.
FINISH_REASONS = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length",
                  "stop_sequence": "stop"}


def _blocks(message: ChatMessage, with_tools: bool) -> list[dict[str, Any]]:
    """One message's content blocks. Empty text blocks are rejected by
    Converse, so they are left out."""
    if message.role == "tool":
        if with_tools:
            return [{"toolResult": {"toolUseId": message.tool_call_id or "",
                                    "content": [{"text": message.content or "(empty)"}]}}]
        return [{"text": f"Result of {message.name or 'the tool'}: {message.content}"}]

    blocks: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
    for call in message.tool_calls:
        if with_tools:
            blocks.append({"toolUse": {"toolUseId": call.id, "name": call.name,
                                       "input": call.arguments}})
        else:
            blocks.append({"text": f"(called {call.name} with {json.dumps(call.arguments)})"})
    return blocks


def to_converse(messages: list[ChatMessage], tools: list[ToolSpec] | None) -> dict[str, Any]:
    """The Converse request body for these messages, minus inference config."""
    system = [{"text": m.content} for m in messages if m.role == "system" and m.content]
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "assistant" if message.role == "assistant" else "user"
        blocks = _blocks(message, with_tools=bool(tools))
        if not blocks:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    body: dict[str, Any] = {"messages": turns}
    if system:
        body["system"] = system
    if tools:
        body["toolConfig"] = {
            "tools": [{"toolSpec": {"name": t.name, "description": t.description,
                                    "inputSchema": {"json": t.parameters}}} for t in tools],
            "toolChoice": {"auto": {}},
        }
    return body


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        settings = get_settings()
        self.region = settings.bedrock_region
        self._model = settings.bedrock_model_id
        self.timeout = settings.llm_timeout_seconds
        self.base_url = f"https://bedrock-runtime.{self.region}.amazonaws.com"

        auth: httpx.Auth | None = None
        headers: dict[str, str] = {}
        if settings.aws_bearer_token_bedrock:
            headers["Authorization"] = f"Bearer {settings.aws_bearer_token_bedrock}"
            self.credentials = "Bedrock API key"
        elif settings.aws_access_key_id and settings.aws_secret_access_key:
            auth = SigV4Auth(settings.aws_access_key_id, settings.aws_secret_access_key,
                             self.region, "bedrock", settings.aws_session_token)
            self.credentials = "IAM credentials"
        else:
            self.credentials = ""
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                         headers=headers, auth=auth, transport=transport)

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    def _path(self, action: str) -> str:
        # Model ids contain ":" (e.g. amazon.nova-lite-v1:0); encode it.
        return f"/model/{quote(self._model, safe='')}/{action}"

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.credentials:
            raise LLMAuthError(
                "LLM_PROVIDER=bedrock but no AWS credentials are set. Set "
                "AWS_BEARER_TOKEN_BEDROCK, or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        payload = to_converse(messages, tools)
        payload["inferenceConfig"] = {"temperature": temperature}
        if max_tokens:
            payload["inferenceConfig"]["maxTokens"] = max_tokens

        started = time.perf_counter()
        try:
            resp = await self._client.post(self._path("converse"), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Bedrock ({self.region}) timed out after "
                                  f"{self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Cannot reach {self.base_url}: {exc}") from exc

        if resp.status_code in (401, 403):
            raise LLMAuthError(f"Bedrock rejected the credentials ({resp.status_code}): "
                               f"{resp.text[:200]}")
        if resp.status_code in (429, 503):
            raise LLMUnavailableError(f"Bedrock is throttling or unavailable "
                                      f"({resp.status_code}). Retry shortly.")
        if resp.status_code >= 400:
            raise LLMBadResponseError(f"Bedrock returned {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            content = data["output"]["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMBadResponseError("Unparseable response from Bedrock") from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in content:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                args = use.get("input")
                calls.append(ToolCall(id=use.get("toolUseId") or f"call_{len(calls)}",
                                      name=use.get("name", ""),
                                      arguments=args if isinstance(args, dict) else {}))

        usage = data.get("usage") or {}
        stop = data.get("stopReason")
        return LLMResponse(
            content="".join(text_parts).strip(),
            tool_calls=calls,
            provider=self.name,
            model=self._model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.get("inputTokens"),
            completion_tokens=usage.get("outputTokens"),
            finish_reason=FINISH_REASONS.get(stop, stop),
        )

    async def health(self) -> ProviderHealth:
        """Configuration only, deliberately: Bedrock has no free endpoint on
        the runtime host, and a probe through Converse would spend tokens on
        every health check. A bad key or model id surfaces on the first chat
        as a typed auth or bad-response error."""
        if not self.credentials:
            return ProviderHealth(healthy=False, provider=self.name, model=self._model,
                                  detail="no AWS credentials set")
        return ProviderHealth(healthy=True, provider=self.name, model=self._model,
                              detail=f"configured ({self.credentials}, {self.region}); "
                                     "not probed")
