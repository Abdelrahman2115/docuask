"""Retrievers: given a question, return the top-k chunks with scores.

Every retriever has the same `retrieve(query, k)` method, so the RAG pipeline
and the evaluation don't care which one they get. Phase 6 adds BM25, hybrid
and reranking retrievers next to this baseline.
"""

from __future__ import annotations

from typing import Protocol

from docuask.config import Settings, get_settings
from docuask.index import EmbedderLike, SearchHit, collection_name, open_collection, search


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[SearchHit]: ...


class VectorRetriever:
    """Baseline: cosine-similarity top-k over the Chroma index."""

    def __init__(self, embedder: EmbedderLike, collection):
        self.embedder = embedder
        self.collection = collection
        self.name = f"vector[{embedder.model_name.split('/')[-1]}]"

    def retrieve(self, query: str, k: int) -> list[SearchHit]:
        return search(self.collection, self.embedder, query, k=k)


def load_vector_retriever(settings: Settings | None = None, model_name: str | None = None) -> VectorRetriever:
    from docuask.embeddings import Embedder

    s = settings or get_settings()
    model = model_name or s.embeddings.model
    col = open_collection(s.paths.index_dir, collection_name(s.vector_store.collection, model))
    return VectorRetriever(Embedder(model), col)
