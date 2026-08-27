"""OpenAI-compatible cloud provider.

Deliberately not an OpenAI-specific client. The `/v1/chat/completions` contract
is the de-facto standard, so one adapter plus a base URL covers:

    Hugging Face router   https://router.huggingface.co/v1     (open-source models)
    OpenAI                https://api.openai.com/v1
    Groq                  https://api.groq.com/openai/v1
    OpenRouter            https://openrouter.ai/api/v1
    vLLM / LM Studio      http://your-host/v1

That is what makes the model swap a one-line `.env` change rather than a code
change, which is the requirement in section 3.2 of the brief.
"""

from __future__ import annotations

import json
import time
from typing import Any

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
from app.logging import get_logger

log = get_logger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout = timeout or settings.llm_timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )

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
        if not self.api_key:
            raise LLMAuthError(
                "LLM_API_KEY is empty but LLM_PROVIDER=openai_compat. "
                "Set a key, or switch LLM_PROVIDER=ollama to run fully locally."
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"{self.base_url} timed out after {self.timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Cannot reach {self.base_url}: {exc}") from exc

        if resp.status_code in (401, 403):
            raise LLMAuthError(f"Provider rejected the API key ({resp.status_code}).")
        if resp.status_code == 429:
            raise LLMUnavailableError("Provider rate-limited this request (429). Retry shortly.")
        if resp.status_code >= 400:
            raise LLMBadResponseError(f"Provider returned {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMBadResponseError(f"Unparseable response from {self.base_url}") from exc

        calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                log.warning("tool_arguments_unparseable", tool=fn.get("name"), raw=raw_args[:200])
                args = {}
            calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{len(calls)}",
                    name=fn.get("name", ""),
                    arguments=args if isinstance(args, dict) else {},
                )
            )

        usage = data.get("usage") or {}
        return LLMResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            provider=self.name,
            model=self._model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )

    async def health(self) -> ProviderHealth:
        if not self.api_key:
            return ProviderHealth(
                healthy=False,
                provider=self.name,
                model=self._model,
                detail="LLM_API_KEY not set",
            )
        started = time.perf_counter()
        try:
            resp = await self._client.get("/models", timeout=8)
            reachable = resp.status_code < 500
            detail = "ok" if reachable else f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 - health probes never raise
            return ProviderHealth(
                healthy=False,
                provider=self.name,
                model=self._model,
                detail=f"unreachable: {type(exc).__name__}",
            )
        return ProviderHealth(
            healthy=reachable,
            provider=self.name,
            model=self._model,
            detail=detail,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
