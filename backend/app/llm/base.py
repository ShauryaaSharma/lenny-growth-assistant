"""Provider-agnostic LLM interface.

This is the seam that makes the model swappable at runtime. Everything above it
-- the agent loop, the skills, the API -- speaks only these types. Adding a new
provider means implementing `LLMProvider` and registering it; no caller changes.

Note on the assignment's suggested agent SDKs: the brief names the Claude Agent
SDK or Pi Coding Agent, but also requires the submitted demo to run on local
Ollama. Neither SDK executes against an Ollama endpoint, so building the agent
directly on one would leave the mandatory demo path unimplementable. We
therefore define this interface and implement the agent loop against it, which
keeps a single identical code path across local and cloud providers. The cost is
that we re-implement session/tool plumbing the SDKs would have supplied; that
trade-off is documented in docs/architecture.md.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(Exception):
    """Base class for provider failures surfaced to the API as typed errors."""

    code = "llm_error"
    http_status = 502


class LLMUnavailableError(LLMError):
    """Provider unreachable -- Ollama not running, DNS failure, connection refused."""

    code = "llm_unavailable"
    http_status = 503


class LLMTimeoutError(LLMError):
    """Provider accepted the request but did not answer in time."""

    code = "llm_timeout"
    http_status = 504


class LLMAuthError(LLMError):
    """Missing or rejected API key."""

    code = "llm_auth"
    http_status = 502


class LLMBadResponseError(LLMError):
    """Provider replied with something we cannot parse."""

    code = "llm_bad_response"
    http_status = 502


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ProviderHealth:
    healthy: bool
    provider: str
    model: str
    detail: str = ""
    models_available: list[str] = field(default_factory=list)
    latency_ms: int | None = None


class LLMProvider(ABC):
    """Every provider implements exactly this."""

    name: str

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One completion. Raises an LLMError subclass on failure."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Cheap reachability probe. Must never raise."""

    async def aclose(self) -> None:
        return None
