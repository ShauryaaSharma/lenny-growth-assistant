"""Ollama provider -- the local, no-API-key path used for the required demo.

Uses Ollama's native `/api/chat`, which supports tool definitions directly.

The one wrinkle worth knowing about: small quantised models (3B-8B) are
inconsistent tool callers. They frequently emit a well-formed tool call as
*plain text in the content field* instead of populating `tool_calls`. Rather
than let that surface as the assistant babbling JSON at the user, we salvage it
-- see `_salvage_tool_call`. This is the single biggest reliability difference
between the local and cloud paths on this stack.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import httpx

from app.config import get_settings
from app.llm.base import (
    ChatMessage,
    LLMBadResponseError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderHealth,
    ToolCall,
    ToolSpec,
)
from app.logging import get_logger

log = get_logger(__name__)

# Matches a fenced or bare JSON object that looks like a tool call.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{[^{}]*\"name\"[^{}]*\})", re.DOTALL)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self, base_url: str | None = None, model: str | None = None, timeout: int | None = None
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_seconds
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_ollama_message(m) for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]

        started = time.perf_counter()
        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Ollama did not respond within {self.timeout}s. "
                f"Large models on CPU can exceed this; raise OLLAMA_TIMEOUT_SECONDS "
                f"or switch to a smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url}. Is `ollama serve` running? ({exc})"
            ) from exc

        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self._model}' is not present in Ollama. Run: ollama pull {self._model}"
            )
        if resp.status_code >= 400:
            raise LLMBadResponseError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMBadResponseError("Ollama returned non-JSON output") from exc

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=(tc.get("function") or {}).get("name", ""),
                arguments=self._coerce_args((tc.get("function") or {}).get("arguments")),
            )
            for tc in (message.get("tool_calls") or [])
        ]

        if not calls and tools:
            salvaged = self._salvage_tool_call(content, tools)
            if salvaged is not None:
                log.info("ollama_tool_call_salvaged", tool=salvaged.name, model=self._model)
                calls = [salvaged]
                content = ""

        return LLMResponse(
            content=content,
            tool_calls=calls,
            provider=self.name,
            model=self._model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )

    @staticmethod
    def _to_ollama_message(m: ChatMessage) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
            ]
        # Ollama has no tool_call_id concept; the tool name carries the linkage.
        if m.role == "tool" and m.name:
            msg["name"] = m.name
        return msg

    @staticmethod
    def _coerce_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _salvage_tool_call(content: str, tools: list[ToolSpec]) -> ToolCall | None:
        """Recover a tool call a small model emitted as text rather than structure.

        Only accepts a payload naming a tool we actually registered, so a model
        hallucinating a function name still falls through to a normal answer.
        """
        if not content or "{" not in content:
            return None
        valid = {t.name for t in tools}
        for match in _JSON_BLOCK_RE.finditer(content):
            blob = match.group(1) or match.group(2)
            if not blob:
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
            if name not in valid:
                continue
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args if isinstance(args, dict) else {},
            )
        return None

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            resp = await self._client.get("/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001 - health probes never raise
            return ProviderHealth(
                healthy=False,
                provider=self.name,
                model=self._model,
                detail=f"unreachable at {self.base_url}: {type(exc).__name__}",
            )

        # Ollama reports "llama3.2:3b"; accept a bare "llama3.2" config too.
        present = any(n == self._model or n.split(":")[0] == self._model for n in names)
        return ProviderHealth(
            healthy=present,
            provider=self.name,
            model=self._model,
            detail="ok" if present else f"model '{self._model}' not pulled",
            models_available=names,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
