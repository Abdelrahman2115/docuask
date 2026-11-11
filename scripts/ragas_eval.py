"""Phase 5: score a finished run with RAGAS (LLM-as-judge).

    python scripts/ragas_eval.py --name baseline
    python scripts/ragas_eval.py --name baseline --judge qwen/qwen3.8-27b --mode MD_JSON

Reads data/eval/runs/<name>/answers.jsonl (from run_eval.py) and scores every
answerable question that was answered (not refused):
    faithfulness      - are the answer's claims supported by the retrieved passages?
    answer_relevancy  - does the answer address the question?
Context precision/recall are computed exactly from gold citations in run_eval.py,
so they don't spend judge quota here.

Scores go to ragas_scores.jsonl (resumable) and are merged into metrics.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from statistics import mean

from docuask.config import PROJECT_ROOT
from docuask.judge import default_judge, make_judge, make_judge_embeddings, strip_citations  # patches ragas

RUNS = PROJECT_ROOT / "data" / "eval" / "runs"


async def score_row(row, faith, relev):
    response = strip_citations(row["answer"])
    out = {"id": row["id"]}
    for name, coro in [
        ("faithfulness", lambda: faith.ascore(user_input=row["question"], response=response,
                                               retrieved_contexts=row["contexts"])),
        ("answer_relevancy", lambda: relev.ascore(user_input=row["question"], response=response)),
    ]:
        try:
            out[name] = float((await coro()).value)
        except Exception as e:  # noqa: BLE001 - record and continue
            out[name] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    judge_model, judge_mode = default_judge()       # from data/eval/judge.json if probe_eval.py ran
    p.add_argument("--judge", default=judge_model)
    p.add_argument("--mode", default=judge_mode, help="instructor mode: JSON, MD_JSON or TOOLS")
    p.add_argument("--strictness", type=int, default=1,
                   help="questions generated per answer for answer_relevancy (ragas default 3; 1 saves quota)")
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between questions (rate limits)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retry-failed", action="store_true")
    args = p.parse_args()

    run_dir = RUNS / args.name
    rows = [json.loads(l) for l in (run_dir / "answers.jsonl").read_text("utf-8").splitlines() if l.strip()]
    todo = [r for r in rows if r["answerable"] and "answer" in r and not r["refused"]][: args.limit]

    score_path = run_dir / "ragas_scores.jsonl"
    scores = {}
    if score_path.exists():
        for line in score_path.read_text("utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                scores[d["id"]] = d
    if args.retry_failed:
        scores = {k: v for k, v in scores.items()
                  if v.get("faithfulness") is not None and v.get("answer_relevancy") is not None}

    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    llm = make_judge(args.judge, args.mode)
    faith = Faithfulness(llm=llm)
    relev = AnswerRelevancy(llm=llm, embeddings=make_judge_embeddings(), strictness=args.strictness)
    print(f"Judge: {args.judge} (mode={args.mode}, strictness={args.strictness}); scoring {len(todo)} answered questions")

    async def run_all():
        # one event loop for the whole run: the async Groq client must not outlive its loop
        for i, row in enumerate(todo, 1):
            if row["id"] in scores:
                continue
            t0 = time.perf_counter()
            res = await score_row(row, faith, relev)
            res["judge"] = args.judge
            scores[row["id"]] = res
            errs = [k for k in res if k.endswith("_error")]
            print(f"[{i:2d}/{len(todo)}] {row['id']} faithfulness={res['faithfulness']} "
                  f"relevancy={res['answer_relevancy']} ({time.perf_counter() - t0:.1f}s)"
                  + (f"  errors: {errs} {res[errs[0]]}" if errs else ""), flush=True)
            with open(score_path, "a", encoding="utf-8") as f:   # save progress after every question
                f.write(json.dumps(res) + "\n")
            await asyncio.sleep(args.sleep)

    asyncio.run(run_all())

    with open(score_path, "w", encoding="utf-8") as f:
        for d in scores.values():
            f.write(json.dumps(d) + "\n")

    def agg(key):
        vals = [d[key] for d in scores.values() if d.get(key) is not None]
        return {"mean": round(mean(vals), 4) if vals else None, "valid": f"{len(vals)}/{len(todo)}"}

    summary = {"judge": args.judge, "mode": args.mode, "answer_relevancy_strictness": args.strictness,
               "faithfulness": agg("faithfulness"), "answer_relevancy": agg("answer_relevancy")}
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else {}
    metrics["ragas"] = summary
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n===== RAGAS =====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
