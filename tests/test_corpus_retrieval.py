"""End-to-end checks on the real corpus (skipped until scripts/ingest.py has run).

These complement the unit tests: they catch regressions in cleaning/chunking
that only show up on the actual FCC documents.
"""

import pytest

from docuask.config import load_settings
from docuask.ingest import load_chunks
from docuask.variants import chunks_path, overlap_for

S = load_settings()
PATH = chunks_path(S)
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="run scripts/ingest.py first")


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(PATH)


def test_corpus_has_all_documents(chunks):
    assert len({c.doc_id for c in chunks}) == 40


def test_chunks_are_clean_and_sized(chunks):
    limit = S.chunking.chunk_size + 100                      # + merged tiny fragment
    assert all(100 <= len(c.text) <= limit for c in chunks)
    assert not any("·" in c.text for c in chunks)       # letterhead removed
    assert not any("fcc504@fcc.gov" in c.text for c in chunks)  # 'Alternate formats' boilerplate removed


def test_every_chunk_has_exactly_one_location(chunks):
    for c in chunks:
        assert (c.page is None) != (c.section is None)
        assert c.citation.startswith(f"[{c.source_file}, ")


def test_overlap_rule():
    assert overlap_for(800) == 120 and overlap_for(400) == 60 and overlap_for(1200) == 180


@pytest.mark.parametrize("question, doc", [
    ("keep my phone number when I change providers", "porting"),
    ("unauthorized charges on my telephone bill cramming", "telephone_bill"),
    ("one ring scam international call back", "one_ring_scam"),
    ("text to 911 emergency", "text_to_911"),
])
def test_bm25_finds_the_right_guide(chunks, question, doc):
    pytest.importorskip("rank_bm25")
    from docuask.retriever import BM25Retriever

    hits = BM25Retriever(chunks).retrieve(question, 4)
    assert doc in {h.chunk.doc_id for h in hits}
