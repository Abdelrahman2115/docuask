"""Retrievers: given a question, return the top-k chunks with scores.

Every retriever has the same `retrieve(query, k)` method and a `name`, so the
RAG pipeline, the app and the evaluation never care which one they get.

    VectorRetriever   dense embeddings, cosine similarity (baseline)
    BM25Retriever     keyword matching (Okapi BM25)
    HybridRetriever   BM25 + vectors merged with Reciprocal Rank Fusion
    RerankRetriever   wraps any retriever: takes N candidates, re-scores each
                      (question, passage) pair with a cross-encoder, keeps top k
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from docuask.config import Settings, get_settings
from docuask.index import EmbedderLike, SearchHit, open_collection, search
from docuask.ingest import Chunk, load_chunks
from docuask.variants import chunks_path, index_collection


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[SearchHit]: ...


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #
class VectorRetriever:
    """Baseline: cosine-similarity top-k over the Chroma index."""

    def __init__(self, embedder: EmbedderLike, collection):
        self.embedder = embedder
        self.collection = collection
        self.name = f"vector[{embedder.model_name.split('/')[-1]}]"

    def retrieve(self, query: str, k: int) -> list[SearchHit]:
        return search(self.collection, self.embedder, query, k=k)


# --------------------------------------------------------------------------- #
# Sparse
# --------------------------------------------------------------------------- #
_STOPWORDS = set("""a an and are as at be but by can do does for from has have how i if in into is it
its me my no not of on or our so that the their them then there these they this to was we were
what when where which who why will with you your yours""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords. Keeps numbers ('911', '25') and
    hyphenated terms split into parts ('text-to-911' -> text, 911)."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Keyword retrieval. Strong on exact terms (names, numbers, jargon like 'cramming')
    that dense embeddings sometimes blur."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        # title + text, the same text the dense index embeds
        self.bm25 = BM25Okapi([tokenize(f"{c.title}\n{c.text}") for c in chunks])
        self.name = "bm25"

    def retrieve(self, query: str, k: int) -> list[SearchHit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: list[list[SearchHit]], rrf_k: int = 60) -> list[SearchHit]:
    """Merge ranked lists: score(d) = sum over lists of 1 / (rrf_k + rank(d)).

    RRF uses only ranks, so it needs no tuning to combine BM25 scores (unbounded)
    with cosine similarities (-1..1). rrf_k=60 is the value from the original
    paper (Cormack et al., 2009) and is the common default.
    """
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunk_by_id[cid] = hit.chunk
    order = sorted(fused, key=fused.get, reverse=True)
    return [SearchHit(chunk_by_id[cid], fused[cid]) for cid in order]


class HybridRetriever:
    def __init__(self, dense: VectorRetriever, sparse: BM25Retriever, candidates: int = 20, rrf_k: int = 60):
        self.dense, self.sparse = dense, sparse
        self.candidates, self.rrf_k = candidates, rrf_k
        self.name = f"hybrid[{dense.name}+bm25]"

    def retrieve(self, query: str, k: int) -> list[SearchHit]:
        n = max(k, self.candidates)
        fused = reciprocal_rank_fusion(
            [self.dense.retrieve(query, n), self.sparse.retrieve(query, n)], self.rrf_k)
        return fused[:k]


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
class RerankRetriever:
    """Two-stage retrieval: a fast retriever proposes candidates, a cross-encoder
    reads each (question, passage) pair together and re-scores it.

    A bi-encoder (the embedding model) encodes question and passage separately,
    so it can't model how specific words interact; a cross-encoder can, which is
    more accurate but too slow to run over the whole corpus - hence two stages.
    """

    def __init__(self, base, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", candidates: int = 20):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.model = CrossEncoder(model_name, device="cpu")
        self.candidates = candidates
        self.name = f"{base.name}+rerank[{model_name.split('/')[-1]}]"

    def retrieve(self, query: str, k: int) -> list[SearchHit]:
        cands = self.base.retrieve(query, max(k, self.candidates))
        if not cands:
            return []
        scores = self.model.predict([(query, f"{h.chunk.title}\n{h.chunk.text}") for h in cands],
                                    show_progress_bar=False)
        order = np.argsort(np.asarray(scores))[::-1][:k]
        return [SearchHit(cands[i].chunk, float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_retriever(s: Settings | None = None, *, mode: str | None = None, model: str | None = None,
                    chunk_size: int | None = None, rerank: bool | None = None, embedder=None):
    """Build the retriever described by config, with optional overrides (used by Phase 6)."""
    s = s or get_settings()
    mode = mode or s.retrieval.mode
    model = model or s.embeddings.model
    rerank = s.retrieval.rerank if rerank is None else rerank

    dense = sparse = None
    if mode in ("vector", "hybrid"):
        if embedder is None:
            from docuask.embeddings import Embedder

            embedder = Embedder(model)
        dense = VectorRetriever(embedder, open_collection(s.paths.index_dir, index_collection(s, model, chunk_size)))
    if mode in ("bm25", "hybrid"):
        path = chunks_path(s, chunk_size)
        if not path.exists():
            raise RuntimeError(f"{path.name} not found. Run: python scripts/ingest.py --chunk-size {chunk_size}")
        sparse = BM25Retriever(load_chunks(path))

    if mode == "vector":
        retriever = dense
    elif mode == "bm25":
        retriever = sparse
    elif mode == "hybrid":
        retriever = HybridRetriever(dense, sparse, s.retrieval.candidates, s.retrieval.rrf_k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    if rerank:
        retriever = RerankRetriever(retriever, s.retrieval.reranker, s.retrieval.candidates)
    return retriever


def load_vector_retriever(settings: Settings | None = None, model_name: str | None = None) -> VectorRetriever:
    """Kept for backward compatibility with Phase 4 scripts."""
    return build_retriever(settings, mode="vector", model=model_name, rerank=False)
