"""Provider selection and fallback.

The registry is the only place that knows which providers exist. Callers ask for
"the active provider" and get whatever `.env` selected, so the model toggle never
leaks into application logic.

Fallback policy: if `LLM_FALLBACK_PROVIDER` is set and the primary raises an
availability/timeout error mid-request, we retry once on the fallback and label
the response with the provider that actually served it -- the UI badge shows the
truth, not the configured intent. Auth and bad-response errors do *not* trigger
fallback: those are configuration bugs, and silently masking them would make the
system harder to operate, not easier.
"""

from __future__ import annotations

from app.config import get_settings
from app.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    ProviderHealth,
    ToolSpec,
)
from app.llm.ollama import OllamaProvider
from app.llm.openai_compat import OpenAICompatProvider
from app.logging import get_logger

log = get_logger(__name__)

FALLBACK_TRIGGERS = (LLMUnavailableError, LLMTimeoutError)

_providers: dict[str, LLMProvider] = {}


def build_provider(name: str) -> LLMProvider:
    if name == "ollama":
        return OllamaProvider()
    if name == "openai_compat":
        return OpenAICompatProvider()
    raise ValueError(f"Unknown LLM provider '{name}'. Expected 'ollama' or 'openai_compat'.")


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a cached provider instance (clients hold connection pools)."""
    settings = get_settings()
    name = name or settings.llm_provider
    if name not in _providers:
        _providers[name] = build_provider(name)
    return _providers[name]


async def chat_with_fallback(
    messages: list[ChatMessage],
    tools: list[ToolSpec] | None = None,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Run a completion on the active provider, falling back if it is down."""
    settings = get_settings()
    primary = get_provider(settings.llm_provider)

    try:
        return await primary.chat(messages, tools, temperature, max_tokens)
    except FALLBACK_TRIGGERS as exc:
        fallback_name = settings.llm_fallback_provider
        if fallback_name in ("none", "", settings.llm_provider):
            log.error(
                "llm_call_failed_no_fallback",
                provider=settings.llm_provider,
                error=str(exc),
            )
            raise
        log.warning(
            "llm_falling_back",
            primary=settings.llm_provider,
            fallback=fallback_name,
            error=str(exc),
        )
        try:
            return await get_provider(fallback_name).chat(messages, tools, temperature, max_tokens)
        except LLMError as fallback_exc:
            log.error("llm_fallback_also_failed", fallback=fallback_name, error=str(fallback_exc))
            raise


async def health_all() -> list[ProviderHealth]:
    """Probe every configured provider for /health/deep."""
    settings = get_settings()
    names = {settings.llm_provider}
    if settings.llm_fallback_provider not in ("none", ""):
        names.add(settings.llm_fallback_provider)
    return [await get_provider(n).health() for n in sorted(names)]


async def close_all() -> None:
    for provider in _providers.values():
        await provider.aclose()
    _providers.clear()
