import json

import pytest

from docuask.config import PROJECT_ROOT
from docuask.evaluation import generation_checks, load_testset, retrieval_metrics, summarize

A, B, C, D = "[a.pdf, p. 1]", "[b.pdf, p. 2]", "[c.html, § X]", "[d.pdf, p. 3]"


def test_retrieval_perfect_and_miss():
    assert retrieval_metrics([A, B], [A]) == {"hit": True, "hit_at_1": True, "mrr": 1.0,
                                              "context_precision": 1.0, "context_recall": 1.0}
    m = retrieval_metrics([B, C], [A])
    assert m["hit"] is False and m["mrr"] == 0 and m["context_precision"] == 0 and m["context_recall"] == 0


def test_retrieval_rank_matters():
    late = retrieval_metrics([B, C, A], [A])
    assert late["hit"] and not late["hit_at_1"]
    assert late["mrr"] == pytest.approx(1 / 3)
    assert late["context_precision"] == pytest.approx(1 / 3)     # precision@3 at the only relevant rank


def test_recall_all_vs_any():
    assert retrieval_metrics([A, D], [A, B], "all")["context_recall"] == 0.5
    assert retrieval_metrics([A, D], [A, B], "any")["context_recall"] == 1.0


def test_generation_checks():
    ans = {"answerable": True, "gold": [A]}
    una = {"answerable": False, "gold": []}
    g = generation_checks(ans, refused=False, citations=[A], invalid=[])
    assert g["refusal_correct"] and g["cites_gold"] and g["has_citation"]
    assert generation_checks(ans, True, [], [])["false_refusal"]
    assert generation_checks(una, False, [B], [])["missed_refusal"]
    assert generation_checks(una, True, [], [])["refusal_correct"]


def test_summarize_counts():
    rows = [
        {"answerable": True, "hit": True, "mrr": 1.0, "context_precision": 1.0, "context_recall": 1.0,
         "refused": False, "refusal_correct": True, "has_citation": True, "cites_gold": True,
         "n_invalid_citations": 0, "total_ms": 100, "retrieval_ms": 10, "generation_ms": 90},
        {"answerable": False, "refused": True, "refusal_correct": True, "has_citation": False,
         "cites_gold": False, "n_invalid_citations": 0, "total_ms": 50, "retrieval_ms": 10, "generation_ms": 40},
    ]
    s = summarize(rows)
    assert s["retrieval"]["hit_rate"] == 1.0
    assert s["generation"]["refusal_accuracy"] == 1.0
    assert s["generation"]["correct_refusals"] == "1/1"


def test_real_testset_is_well_formed():
    items = load_testset(PROJECT_ROOT / "data" / "eval" / "testset.jsonl")
    assert len(items) == 40 and len({i["id"] for i in items}) == 40
    for i in items:
        assert i["question"].strip() and i["reference"].strip()
        assert bool(i["gold"]) == i["answerable"]
        assert i.get("gold_mode", "all") in {"all", "any"}


def test_gold_citations_exist_in_corpus():
    from docuask.config import load_settings
    from docuask.variants import chunks_path as variant_chunks_path

    chunks_path = variant_chunks_path(load_settings())
    if not chunks_path.exists():
        pytest.skip("run scripts/ingest.py first")
    from docuask.ingest import load_chunks

    corpus = {c.citation for c in load_chunks(chunks_path)}
    items = load_testset(PROJECT_ROOT / "data" / "eval" / "testset.jsonl")
    missing = [(i["id"], g) for i in items for g in i["gold"] if g not in corpus]
    assert missing == []


def test_strip_citations_for_judge():
    pytest.importorskip("langchain_community")
    from docuask.judge import strip_citations

    text = "You can port in one day [porting.pdf, p. 1]. Fees vary [bill.html, § Cramming]."
    assert strip_citations(text) == "You can port in one day . Fees vary ."
