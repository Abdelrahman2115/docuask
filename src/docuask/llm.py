"""Return a LangChain chat model chosen by config, so the LLM is swappable."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from docuask.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    cfg = (settings or get_settings()).llm

    if cfg.provider == "groq":
        if not cfg.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=cfg.groq_api_key,
        )

    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.model,
            temperature=cfg.temperature,
            num_predict=cfg.max_tokens,
            base_url=cfg.ollama_base_url,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
