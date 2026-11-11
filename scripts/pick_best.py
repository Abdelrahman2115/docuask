"""Pick the final retrieval configuration with a rule fixed in advance.

Rule (decided before looking at the combined results):
  1. Highest MRR on the 30 answerable dev questions.
  2. Configurations within one question's worth of MRR of the best
     (0.5 / 30 = 0.017: one question moving from rank 1 to rank 2) count as tied,
     because a smaller difference is noise on 30 questions.
  3. Among tied configurations, pick the lowest latency per query.

Writes data/eval/phase6_choice.json, which run_eval.py --from-choice reads.
"""

from __future__ import annotations

import json
import sys

from docuask.config import PROJECT_ROOT

EVAL = PROJECT_ROOT / "data" / "eval"


def main() -> int:
    results = json.loads((EVAL / "phase6_retrieval.json").read_text("utf-8"))
    n = 30
    tol = 0.5 / n
    best_mrr = max(r["mrr"] for r in results)
    tied = [r for r in results if r["mrr"] >= best_mrr - tol]
    choice = min(tied, key=lambda r: r["mean_retrieval_ms"])

    print(f"Best MRR {best_mrr:.3f}; tolerance {tol:.3f}; tied configurations:")
    for r in sorted(tied, key=lambda r: -r["mrr"]):
        mark = "  <- chosen" if r is choice else ""
        print(f"  {r['name']:20s} mrr={r['mrr']:.3f} hit@1={r['hit_at_1']:.3f} "
              f"ms/query={r['mean_retrieval_ms']:.0f}{mark}")
    out = {k: choice[k] for k in ("name", "mode", "model", "chunk_size", "rerank")}
    out.update({"mrr": choice["mrr"], "hit_at_1": choice["hit_at_1"], "rule": __doc__.split("Writes")[0].strip()})
    (EVAL / "phase6_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nChosen: {out['name']} -> data/eval/phase6_choice.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
