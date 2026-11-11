"""Deterministic evaluation metrics (no LLM judge needed).

Retrieval metrics use the gold citations in data/eval/testset.jsonl: a retrieved
chunk is relevant if its citation ([file, p. N] or [file, § Section]) is gold.
This is the "ID-based" style of context precision/recall: cheap, exact, and
repeatable, so Phase 6 can compare many retrievers without spending API quota.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def load_testset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def retrieval_metrics(retrieved: list[str], gold: list[str], gold_mode: str = "all") -> dict:
    """retrieved: citations in rank order. gold: relevant citations.
    gold_mode 'all' = every gold source is needed; 'any' = any one is enough."""
    rel = [c in gold for c in retrieved]
    hit = any(rel)
    first = next((i for i, r in enumerate(rel, 1) if r), None)

    # Rank-aware context precision (same formula as RAGAS context precision):
    # average of precision@i over the ranks i that hold a relevant chunk.
    precisions = [sum(rel[:i]) / i for i, r in enumerate(rel, 1) if r]
    context_precision = mean(precisions) if precisions else 0.0

    found = {c for c in retrieved if c in gold}
    if gold_mode == "any":
        context_recall = 1.0 if found else 0.0
    else:
        context_recall = len(found) / len(set(gold)) if gold else 0.0

    return {
        "hit": hit,
        "hit_at_1": bool(rel and rel[0]),
        "mrr": 1.0 / first if first else 0.0,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def run_retrieval(retriever, items: list[dict], k: int) -> list[dict]:
    """Retrieval-only evaluation rows (no LLM): what run_eval.py --retrieval-only writes."""
    import time

    rows = []
    for item in items:
        t0 = time.perf_counter()
        hits = retriever.retrieve(item["question"], k)
        row = {"id": item["id"], "answerable": item["answerable"], "type": item["type"],
               "question": item["question"], "retrieval_ms": 1000 * (time.perf_counter() - t0),
               "retrieved": [h.chunk.citation for h in hits], "contexts": [h.chunk.text for h in hits],
               "scores": [round(h.score, 4) for h in hits]}
        if item["answerable"]:
            row.update(retrieval_metrics(row["retrieved"], item["gold"], item.get("gold_mode", "all")))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Generation (rule-based)
# --------------------------------------------------------------------------- #
def generation_checks(item: dict, refused: bool, citations: list[str], invalid: list[str]) -> dict:
    answerable = item["answerable"]
    return {
        "refusal_correct": (refused != answerable),          # refused iff unanswerable
        "false_refusal": answerable and refused,
        "missed_refusal": (not answerable) and (not refused),  # answered an unanswerable question
        "has_citation": bool(citations),
        "cites_gold": any(c in item["gold"] for c in citations),
        "n_invalid_citations": len(invalid),
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _avg(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else None


def summarize(rows: list[dict]) -> dict:
    """rows: one dict per question as written by scripts/run_eval.py."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    out = {
        "n_questions": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "retrieval": {
            "hit_at_1": _avg(ans, "hit_at_1"),
            "hit_rate": _avg(ans, "hit"),
            "mrr": _avg(ans, "mrr"),
            "context_precision": _avg(ans, "context_precision"),
            "context_recall": _avg(ans, "context_recall"),
            "mean_context_chars": _avg([{"n": sum(len(c) for c in r.get("contexts", []))} for r in ans], "n"),
            "mean_retrieval_ms": _avg(rows, "retrieval_ms"),
        },
    }
    if rows and "refused" in rows[0]:
        answered = [r for r in ans if not r["refused"]]
        out["generation"] = {
            "refusal_accuracy": _avg(rows, "refusal_correct"),
            "correct_refusals": f"{sum(r['refused'] for r in una)}/{len(una)}",
            "false_refusals": f"{sum(r['refused'] for r in ans)}/{len(ans)}",
            "answers_with_citation": _avg(answered, "has_citation"),
            "answers_citing_gold_source": _avg(answered, "cites_gold"),
            "invalid_citations_total": sum(r["n_invalid_citations"] for r in rows),
            "mean_total_ms": _avg(rows, "total_ms"),
            "mean_retrieval_ms": _avg(rows, "retrieval_ms"),
            "mean_generation_ms": _avg([r for r in rows if not r.get("skipped_llm")], "generation_ms"),
        }
    return out
