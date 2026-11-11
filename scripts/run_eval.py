"""Phase 5: run the test set through the pipeline and compute deterministic metrics.

    python scripts/run_eval.py --name baseline                 # retrieval + generation
    python scripts/run_eval.py --name baseline --retrieval-only  # no LLM calls (fast, free)

Results go to data/eval/runs/<name>/:
    answers.jsonl   one line per question (resumable: finished ids are skipped)
    metrics.json    summary numbers
Re-run with --fresh to start over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

from docuask.config import PROJECT_ROOT, get_settings
from docuask.evaluation import generation_checks, load_testset, retrieval_metrics, summarize
from docuask.prompts import PROMPT_VERSION
from docuask.retriever import build_retriever

TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"
RUNS = PROJECT_ROOT / "data" / "eval" / "runs"


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    p.add_argument("--k", type=int, default=s.retrieval.top_k)
    p.add_argument("--model", default=s.embeddings.model, help="embedding model / index to use")
    p.add_argument("--mode", default=s.retrieval.mode, choices=["vector", "bm25", "hybrid"])
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=s.retrieval.rerank)
    p.add_argument("--chunk-size", type=int, default=s.chunking.chunk_size)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--rpm", type=int, default=None, help="override LLM requests per minute")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    items = load_testset(TESTSET)[: args.limit]
    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "answers.jsonl"
    if args.fresh and out_path.exists():
        out_path.unlink()
    done = {}
    if out_path.exists():
        for line in out_path.read_text("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r

    retriever = build_retriever(s, mode=args.mode, model=args.model, chunk_size=args.chunk_size, rerank=args.rerank)
    pipeline = None
    if not args.retrieval_only:
        from docuask.llm import get_llm
        from docuask.rag import RAGPipeline

        llm_settings = s if args.rpm is None else replace(s, llm=replace(s.llm, requests_per_minute=args.rpm))
        pipeline = RAGPipeline(retriever, get_llm(llm_settings), top_k=args.k, min_score=s.retrieval.min_score)

    config = {
        "name": args.name, "retriever": retriever.name, "k": args.k,
        "embedding_model": args.model, "mode": args.mode, "rerank": args.rerank,
        "chunk_size": args.chunk_size, "min_score": s.retrieval.min_score,
        "llm": None if args.retrieval_only else f"{s.llm.provider}/{s.llm.model}",
        "prompt_version": None if args.retrieval_only else PROMPT_VERSION,
    }
    print("Config:", json.dumps(config))

    with open(out_path, "a", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            row = {"id": item["id"], "answerable": item["answerable"], "type": item["type"],
                   "question": item["question"]}
            if pipeline is None:
                t0 = time.perf_counter()
                hits = retriever.retrieve(item["question"], args.k)
                row["retrieval_ms"] = 1000 * (time.perf_counter() - t0)
            else:
                try:
                    r = pipeline.answer(item["question"])
                except Exception as e:  # noqa: BLE001 - keep going, record the failure
                    print(f"[{i:2d}/{len(items)}] {item['id']} ERROR {type(e).__name__}: {e}")
                    continue
                hits = r.hits
                row.update({
                    "answer": r.answer, "refused": r.refused, "citations": r.citations,
                    "invalid_citations": r.invalid_citations, "skipped_llm": r.skipped_llm,
                    "retrieval_ms": r.retrieval_ms, "generation_ms": r.generation_ms, "total_ms": r.total_ms,
                })
                row.update(generation_checks(item, r.refused, r.citations, r.invalid_citations))
            row["retrieved"] = [h.chunk.citation for h in hits]
            row["contexts"] = [h.chunk.text for h in hits]
            row["scores"] = [round(h.score, 4) for h in hits]
            if item["answerable"]:
                row.update(retrieval_metrics(row["retrieved"], item["gold"], item.get("gold_mode", "all")))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[item["id"]] = row
            status = ("REFUSED" if row.get("refused") else "answered") if pipeline else "retrieved"
            gold = f"hit={row['hit']}" if item["answerable"] else "unanswerable"
            print(f"[{i:2d}/{len(items)}] {item['id']} {status:9s} {gold}")

    rows = [done[it["id"]] for it in items if it["id"] in done]
    metrics = {"config": config, **summarize(rows)}
    missing = [it["id"] for it in items if it["id"] not in done]
    metrics["missing"] = missing
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n===== METRICS =====")
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} questions failed; re-run the same command to retry them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
