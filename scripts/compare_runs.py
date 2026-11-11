"""Side-by-side table of two full evaluation runs (generation + RAGAS).

    python scripts/compare_runs.py baseline final

Writes data/eval/results_<a>_vs_<b>.md
"""

from __future__ import annotations

import json
import sys

from docuask.config import PROJECT_ROOT
from docuask.evaluation import load_testset, retrieval_metrics, summarize

RUNS = PROJECT_ROOT / "data" / "eval" / "runs"

ROWS = [
    ("RAGAS faithfulness", ("ragas", "faithfulness", "mean")),
    ("RAGAS answer relevancy", ("ragas", "answer_relevancy", "mean")),
    ("Hit@1", ("retrieval", "hit_at_1")),
    ("Hit@4", ("retrieval", "hit_rate")),
    ("MRR", ("retrieval", "mrr")),
    ("Context precision (ID-based)", ("retrieval", "context_precision")),
    ("Context recall (ID-based)", ("retrieval", "context_recall")),
    ("Refusal accuracy", ("generation", "refusal_accuracy")),
    ("Correct refusals", ("generation", "correct_refusals")),
    ("False refusals", ("generation", "false_refusals")),
    ("Answers with a citation", ("generation", "answers_with_citation")),
    ("Answers citing the gold source", ("generation", "answers_citing_gold_source")),
    ("Invalid citations (total)", ("generation", "invalid_citations_total")),
    ("Retrieval ms/query", ("retrieval", "mean_retrieval_ms")),
]


def get(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def load_run(name: str) -> dict:
    """Recompute every code-based metric from answers.jsonl with the current formulas,
    so runs scored at different times are compared like for like. RAGAS scores and
    the run config are taken from metrics.json."""
    gold = {i["id"]: i for i in load_testset(PROJECT_ROOT / "data" / "eval" / "testset.jsonl")}
    rows = [json.loads(l) for l in (RUNS / name / "answers.jsonl").read_text("utf-8").splitlines() if l.strip()]
    for r in rows:
        if r["answerable"]:
            item = gold[r["id"]]
            r.update(retrieval_metrics(r["retrieved"], item["gold"], item.get("gold_mode", "all")))
    saved = json.loads((RUNS / name / "metrics.json").read_text("utf-8"))
    return {**summarize(rows), "config": saved.get("config", {}), "ragas": saved.get("ragas")}


def main() -> int:
    a, b = (sys.argv[1:3] + ["baseline", "final"])[:2]
    ma, mb = load_run(a), load_run(b)
    lines = [f"| Metric | {a} | {b} | Change |", "|---|---|---|---|"]
    for label, path in ROWS:
        va, vb = get(ma, path), get(mb, path)
        delta = f"{vb - va:+.3f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
        lines.append(f"| {label} | {fmt(va)} | {fmt(vb)} | {delta} |")
    cfg = lambda m: m.get("config", {})  # noqa: E731
    header = (f"# {a} vs {b}\n\n- {a}: {cfg(ma).get('retriever')}\n- {b}: {cfg(mb).get('retriever')}\n"
              f"- generator: {cfg(mb).get('llm')}, prompt {cfg(mb).get('prompt_version')}\n"
              f"- judge: {get(mb, ('ragas', 'judge'))}\n\n")
    table = "\n".join(lines)
    out = PROJECT_ROOT / "data" / "eval" / f"results_{a}_vs_{b}.md"
    out.write_text(header + table + "\n", encoding="utf-8")
    print(header + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
