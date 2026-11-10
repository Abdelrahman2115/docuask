"""Return a LangChain chat model chosen by config, so the LLM is swappable."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from docuask.config import Settings, get_settings

REASONING_MODEL_PREFIXES = ("openai/gpt-oss", "qwen/")


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    cfg = (settings or get_settings()).llm
    # Client-side throttle: spreads calls out so batch runs (Phase 5) don't hit 429 errors.
    limiter = InMemoryRateLimiter(
        requests_per_second=cfg.requests_per_minute / 60,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    if cfg.provider == "groq":
        if not cfg.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
        from langchain_groq import ChatGroq

        extra = {}
        if cfg.reasoning_effort and cfg.model.startswith(REASONING_MODEL_PREFIXES):
            extra["reasoning_effort"] = cfg.reasoning_effort
        return ChatGroq(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=cfg.groq_api_key,
            max_retries=6,              # retries with backoff on 429 / 5xx
            rate_limiter=limiter,
            **extra,
        )

    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
            base_url=cfg.ollama_base_url,
            rate_limiter=limiter,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
