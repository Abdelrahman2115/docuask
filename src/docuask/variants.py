"""Names for chunk files and vector collections, one per (chunk size, embedding model).

Every experiment variant gets its own files, so nothing is ever overwritten:
    data/processed/chunks_c800.jsonl
    Chroma collection  docuask-c800-all-minilm-l6-v2
"""

from __future__ import annotations

from pathlib import Path

from docuask.config import Settings
from docuask.index import collection_name

OVERLAP_RATIO = 0.15   # overlap = 15% of chunk size in every experiment (800 -> 120)


def overlap_for(chunk_size: int) -> int:
    return int(round(chunk_size * OVERLAP_RATIO))


def chunks_path(s: Settings, chunk_size: int | None = None) -> Path:
    return s.paths.processed_dir / f"chunks_c{chunk_size or s.chunking.chunk_size}.jsonl"


def index_collection(s: Settings, model: str | None = None, chunk_size: int | None = None) -> str:
    base = f"{s.vector_store.collection}-c{chunk_size or s.chunking.chunk_size}"
    return collection_name(base, model or s.embeddings.model)
