"""Tests for BM25, Reciprocal Rank Fusion and hybrid retrieval (no models needed)."""

import pytest

from docuask.index import SearchHit
from docuask.ingest import Chunk
from docuask.retriever import HybridRetriever, reciprocal_rank_fusion, tokenize


def ch(cid, text, title="T"):
    return Chunk(cid, cid.split(":")[0], title, "u", f"{cid.split(':')[0]}.pdf", 1, None, 0, text)


A = ch("a:p1:c0", "You can keep your phone number when you switch providers.", "Porting")
B = ch("b:p1:c0", "Cramming is placing unauthorized charges on your bill.", "Telephone Bill")
C = ch("c:p1:c0", "If a caller asks you to pay with gift cards, it's a scam.", "Robocalls")


def test_tokenize_drops_stopwords_keeps_numbers():
    assert tokenize("How do I text-to-911 in an emergency?") == ["text", "911", "emergency"]


def test_rrf_rewards_agreement():
    l1 = [SearchHit(A, 0.9), SearchHit(B, 0.8), SearchHit(C, 0.1)]
    l2 = [SearchHit(C, 12.0), SearchHit(A, 3.0)]
    fused = reciprocal_rank_fusion([l1, l2], rrf_k=60)
    assert [h.chunk.chunk_id for h in fused][0] == "a:p1:c0"      # ranked high in both lists
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert len(fused) == 3                                          # union, no duplicates


def test_rrf_ignores_raw_score_scale():
    # Huge BM25 scores must not dominate: only ranks matter.
    l1 = [SearchHit(A, 0.5), SearchHit(B, 0.4)]
    l2 = [SearchHit(B, 1000.0), SearchHit(A, 999.0)]
    fused = reciprocal_rank_fusion([l1, l2])
    assert fused[0].score == pytest.approx(fused[1].score)


class ListRetriever:
    def __init__(self, name, hits):
        self.name, self.hits = name, hits

    def retrieve(self, query, k):
        return self.hits[:k]


def test_hybrid_merges_and_cuts_to_k():
    dense = ListRetriever("vector[x]", [SearchHit(A, 0.7), SearchHit(B, 0.6)])
    sparse = ListRetriever("bm25", [SearchHit(C, 9.0), SearchHit(A, 5.0)])
    hits = HybridRetriever(dense, sparse, candidates=10).retrieve("q", 2)
    # A: 1/61 + 1/62 (in both lists), C: 1/61 (top of BM25), B: 1/62
    assert [h.chunk.chunk_id for h in hits] == ["a:p1:c0", "c:p1:c0"]


def test_bm25_finds_exact_keywords():
    pytest.importorskip("rank_bm25")
    from docuask.retriever import BM25Retriever

    r = BM25Retriever([A, B, C])
    assert r.retrieve("caller wants gift cards", 1)[0].chunk is C
    assert r.retrieve("unauthorized charges cramming", 1)[0].chunk is B
