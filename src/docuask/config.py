"""Load settings from config.yaml, with secrets and overrides from .env.

Usage:
    from docuask.config import get_settings
    s = get_settings()
    print(s.llm.model, s.paths.index_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_PROVIDERS = {"groq", "ollama"}


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class Chunking:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class Embeddings:
    model: str
    multilingual_model: str


@dataclass(frozen=True)
class VectorStore:
    backend: str
    collection: str


@dataclass(frozen=True)
class Retrieval:
    top_k: int
    min_score: float = 0.0     # refuse without calling the LLM if the best hit scores below this (0 = off)
    mode: str = "vector"       # vector | bm25 | hybrid
    rerank: bool = False       # re-score candidates with a cross-encoder
    candidates: int = 20       # how many candidates hybrid/rerank look at before cutting to top_k
    rrf_k: int = 60            # Reciprocal Rank Fusion constant
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class LLM:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    groq_api_key: str | None
    ollama_base_url: str
    requests_per_minute: int = 20      # stay under Groq free-tier limits
    reasoning_effort: str | None = "low"  # for reasoning models (gpt-oss); None to leave unset


@dataclass(frozen=True)
class Settings:
    paths: Paths
    chunking: Chunking
    embeddings: Embeddings
    vector_store: VectorStore
    retrieval: Retrieval
    llm: LLM


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(config_path: Path | None = None, env_path: Path | None = None) -> Settings:
    """Build a Settings object. Environment variables override config.yaml."""
    load_dotenv(env_path or PROJECT_ROOT / ".env", override=False)
    config_path = config_path or PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    provider = os.getenv("DOCUASK_LLM_PROVIDER", cfg["llm"]["provider"]).strip().lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"llm.provider must be one of {sorted(VALID_PROVIDERS)}, got '{provider}'")

    chunk = cfg["chunking"]
    if chunk["chunk_overlap"] >= chunk["chunk_size"]:
        raise ValueError("chunking.chunk_overlap must be smaller than chunking.chunk_size")

    return Settings(
        paths=Paths(**{k: _resolve(v) for k, v in cfg["paths"].items()}),
        chunking=Chunking(**chunk),
        embeddings=Embeddings(**cfg["embeddings"]),
        vector_store=VectorStore(**cfg["vector_store"]),
        retrieval=Retrieval(
            top_k=int(cfg["retrieval"]["top_k"]),
            min_score=float(cfg["retrieval"].get("min_score", 0.0)),
            mode=str(cfg["retrieval"].get("mode", "vector")),
            rerank=bool(cfg["retrieval"].get("rerank", False)),
            candidates=int(cfg["retrieval"].get("candidates", 20)),
            rrf_k=int(cfg["retrieval"].get("rrf_k", 60)),
            reranker=str(cfg["retrieval"].get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        ),
        llm=LLM(
            provider=provider,
            model=os.getenv("DOCUASK_LLM_MODEL", cfg["llm"]["model"]),
            temperature=float(cfg["llm"]["temperature"]),
            max_tokens=int(cfg["llm"]["max_tokens"]),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            requests_per_minute=int(cfg["llm"].get("requests_per_minute", 20)),
            reasoning_effort=cfg["llm"].get("reasoning_effort", "low"),
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for app code. Tests should call load_settings() directly."""
    return load_settings()
