"""Central configuration.

Everything the evaluator might want to change lives here and is driven by
environment variables. Swapping the LLM provider, the model, the database, or
the embedding model requires editing `.env` only — never application code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "openai_compat", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    cors_origins: str = "http://localhost:3000"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://lenny:lenny_dev_password@localhost:5432/lenny"

    # --- Provider selection ---
    llm_provider: ProviderName = "ollama"
    llm_fallback_provider: ProviderName = "none"

    # --- Ollama (local) ---
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 180

    # --- OpenAI-compatible cloud (HF router / OpenAI / Groq / OpenRouter) ---
    llm_base_url: str = "https://router.huggingface.co/v1"
    llm_model: str = "Qwen/Qwen3-32B"
    llm_api_key: str = ""
    llm_timeout_seconds: int = 120

    # --- Retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # Must be on a mounted volume; fastembed ignores the HF cache path.
    embedding_cache_dir: str = "/data/models"
    retrieval_top_k: int = 8
    # Absolute cosine-similarity floor for the grounding guard, for
    # bge-small-en-v1.5. Measured, not guessed: `python -m app.evals.run_eval`
    # against the golden set found the original hand-picked value of 0.55 let
    # 80% of out-of-domain questions ("sourdough starter recipe", "explain
    # general relativity") incorrectly ground, because bge-small's cosine
    # similarity clusters conversational English text more tightly than a
    # single eyeballed threshold accounted for. In-domain questions on this
    # corpus measured 0.71-0.81; out-of-domain measured 0.54-0.66 -- a clean
    # ~0.05 gap. 0.69 sits in that gap with margin on both sides. Re-run the
    # eval after any change to the embedding model, chunk size, or this value.
    retrieval_min_similarity: float = 0.69
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 80

    # --- Ingestion ---
    transcripts_repo_url: str = "https://github.com/ChatPRD/lennys-podcast-transcripts.git"
    transcripts_local_path: str = "/data/lennys-transcripts"
    ingest_episode_limit: int = 0
    ingest_on_startup: bool = True

    @field_validator("cors_origins")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def active_model_name(self) -> str:
        """The model string for whichever provider is currently selected."""
        return self.ollama_model if self.llm_provider == "ollama" else self.llm_model

    def describe_provider(self) -> dict[str, object]:
        """Provider metadata surfaced in the UI badge and /health/deep."""
        if self.llm_provider == "ollama":
            endpoint, key_required = self.ollama_base_url, False
        else:
            endpoint, key_required = self.llm_base_url, True
        return {
            "provider": self.llm_provider,
            "model": self.active_model_name,
            "endpoint": endpoint,
            "is_local": self.llm_provider == "ollama",
            "api_key_required": key_required,
            "api_key_present": bool(self.llm_api_key),
            "fallback_provider": self.llm_fallback_provider,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
