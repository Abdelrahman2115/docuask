"""Index tests with a tiny fake embedder, so no model download is needed."""

import hashlib
import re

import numpy as np
import pytest

from docuask.index import collection_name, from_metadata, passage_text, to_metadata
from docuask.ingest import Chunk


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims, L2-normalized. Shared words -> similar vectors."""

    model_name = "fake/bow-64"

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make_chunks():
    return [
        Chunk("porting:p1:c0", "porting", "Porting", "u1", "porting.pdf", 1, None, 0,
              "You can keep your phone number when you switch providers. This is called porting."),
        Chunk("bill:s1:c0", "bill", "Understanding Your Telephone Bill", "u2", "bill.html", None, "Cramming", 0,
              "Cramming is placing unauthorized charges on your telephone bill."),
        Chunk("wea:p2:c0", "wea", "Wireless Emergency Alerts", "u3", "wea.pdf", 2, None, 0,
              "Wireless emergency alerts warn you about severe weather and AMBER alerts."),
    ]


def test_collection_name_is_valid_slug():
    name = collection_name("docuask", "sentence-transformers/all-MiniLM-L6-v2")
    assert name == "docuask-all-minilm-l6-v2"
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", name)


def test_metadata_roundtrip_keeps_none_values():
    for c in make_chunks():
        m = to_metadata(c)
        assert all(v is not None for v in m.values())          # Chroma rejects None
        assert from_metadata(c.chunk_id, c.text, m) == c


def test_passage_text_includes_title():
    c = make_chunks()[0]
    assert passage_text(c).startswith("Porting\n")


def test_build_and_search(tmp_path):
    pytest.importorskip("chromadb")
    from docuask.index import build_index, open_collection, search

    emb = FakeEmbedder()
    build_index(make_chunks(), emb, tmp_path, "test-index")
    col = open_collection(tmp_path, "test-index")
    assert col.count() == 3

    hits = search(col, emb, "keep my phone number when I switch", k=2)
    assert hits[0].chunk.doc_id == "porting"
    assert hits[0].score >= hits[1].score
    assert -1.0 <= hits[1].score <= 1.0 + 1e-6
    assert hits[0].chunk.citation == "[porting.pdf, p. 1]"

    bill = search(col, emb, "unauthorized charges cramming", k=1)[0]
    assert bill.chunk.page is None and bill.chunk.section == "Cramming"


def test_rebuild_does_not_duplicate(tmp_path):
    pytest.importorskip("chromadb")
    from docuask.index import build_index

    emb = FakeEmbedder()
    build_index(make_chunks(), emb, tmp_path, "test-index")
    col = build_index(make_chunks(), emb, tmp_path, "test-index")
    assert col.count() == 3


def test_duplicate_ids_rejected(tmp_path):
    pytest.importorskip("chromadb")
    from docuask.index import build_index

    chunks = make_chunks()
    with pytest.raises(ValueError):
        build_index(chunks + [chunks[0]], FakeEmbedder(), tmp_path, "test-index")
