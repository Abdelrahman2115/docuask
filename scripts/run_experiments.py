"""Phase 6: retrieval experiments, each measured against the baseline.

    python scripts/run_experiments.py                # all experiments
    python scripts/run_experiments.py --only hybrid rerank

Retrieval-only: no LLM calls, so every variant is free to run and repeatable.
Missing chunk files and vector indexes are built automatically.

Writes:
    data/eval/runs/exp_<name>/answers.jsonl + metrics.json   per experiment
    data/eval/phase6_retrieval.json / .md                    results table
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

from docuask.config import PROJECT_ROOT, get_settings
from docuask.evaluation import load_testset, run_retrieval, summarize
from docuask.index import build_index, get_client
from docuask.ingest import chunk_units, load_corpus, save_chunks
from docuask.retriever import build_retriever
from docuask.variants import chunks_path, index_collection, overlap_for

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-small-en-v1.5"
MULTI_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-small"

# name, group, settings. Baseline first; every other row changes ONE thing, except
# the final combinations, which stack the changes that helped.
EXPERIMENTS = [
    ("baseline",          "baseline",   dict(mode="vector", model=MINILM, chunk_size=800, rerank=False)),
    ("chunk400",          "chunking",   dict(mode="vector", model=MINILM, chunk_size=400, rerank=False)),
    ("chunk1200",         "chunking",   dict(mode="vector", model=MINILM, chunk_size=1200, rerank=False)),
    ("emb_bge_small",     "embeddings", dict(mode="vector", model=BGE, chunk_size=800, rerank=False)),
    ("emb_multi_minilm",  "embeddings", dict(mode="vector", model=MULTI_MINILM, chunk_size=800, rerank=False)),
    ("emb_multi_e5",      "embeddings", dict(mode="vector", model=E5, chunk_size=800, rerank=False)),
    ("bm25",              "hybrid",     dict(mode="bm25", model=None, chunk_size=800, rerank=False)),
    ("hybrid",            "hybrid",     dict(mode="hybrid", model=MINILM, chunk_size=800, rerank=False)),
    ("rerank",            "rerank",     dict(mode="vector", model=MINILM, chunk_size=800, rerank=True)),
    ("hybrid_rerank",     "combined",   dict(mode="hybrid", model=MINILM, chunk_size=800, rerank=True)),
    ("hybrid_rerank_bge", "combined",   dict(mode="hybrid", model=BGE, chunk_size=800, rerank=True)),
]

TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"
RUNS = PROJECT_ROOT / "data" / "eval" / "runs"

_embedders: dict = {}


def get_embedder(model):
    from docuask.embeddings import Embedder

    if model not in _embedders:
        _embedders.clear()          # keep one model in RAM at a time (8 GB machine)
        gc.collect()
        t0 = time.perf_counter()
        _embedders[model] = Embedder(model)
        print(f"  loaded {model} (max_seq_length={_embedders[model].max_seq_length}) "
              f"in {time.perf_counter() - t0:.1f}s")
    return _embedders[model]


def ensure_chunks(s, chunk_size):
    path = chunks_path(s, chunk_size)
    if not path.exists():
        units, _ = load_corpus(s.paths.raw_dir, PROJECT_ROOT / "data" / "sources.csv")
        overlap = s.chunking.chunk_overlap if chunk_size == s.chunking.chunk_size else overlap_for(chunk_size)
        save_chunks(chunk_units(units, chunk_size, overlap), path)
        print(f"  built {path.name} (overlap {overlap})")
    return path


def ensure_index(s, model, chunk_size):
    from docuask.ingest import load_chunks

    name = index_collection(s, model, chunk_size)
    existing = {getattr(c, "name", c) for c in get_client(s.paths.index_dir).list_collections()}
    if name not in existing:
        chunks = load_chunks(ensure_chunks(s, chunk_size))
        t0 = time.perf_counter()
        build_index(chunks, get_embedder(model), s.paths.index_dir, name)
        print(f"  built index {name}: {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")


def fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="experiment names (baseline is always included)")
    p.add_argument("--k", type=int, default=s.retrieval.top_k)
    args = p.parse_args()

    items = load_testset(TESTSET)
    todo = [e for e in EXPERIMENTS if not args.only or e[0] in args.only or e[0] == "baseline"]
    results = []

    for name, group, cfg in todo:
        print(f"\n=== {name} ({group}) {cfg}")
        ensure_chunks(s, cfg["chunk_size"])
        if cfg["mode"] in ("vector", "hybrid"):
            ensure_index(s, cfg["model"], cfg["chunk_size"])
        embedder = get_embedder(cfg["model"]) if cfg["model"] and cfg["mode"] != "bm25" else None
        retriever = build_retriever(s, mode=cfg["mode"], model=cfg["model"], chunk_size=cfg["chunk_size"],
                                    rerank=cfg["rerank"], embedder=embedder)
        retriever.retrieve("warm-up query about phone bills", args.k)   # exclude model warm-up from latency
        rows = run_retrieval(retriever, items, args.k)

        run_dir = RUNS / f"exp_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "answers.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = summarize(rows)
        meta = {"name": name, "group": group, "retriever": retriever.name, "k": args.k, **cfg,
                "max_seq_length": embedder.max_seq_length if embedder else None,
                "n_chunks": sum(1 for _ in open(chunks_path(s, cfg["chunk_size"]), encoding="utf-8"))}
        (run_dir / "metrics.json").write_text(json.dumps({"config": meta, **summary}, indent=2), encoding="utf-8")
        results.append({**meta, **summary["retrieval"],
                         "misses": [r["id"] for r in rows if r["answerable"] and not r["hit"]]})
        r = summary["retrieval"]
        print(f"  hit@1={fmt(r['hit_at_1'])} hit@{args.k}={fmt(r['hit_rate'])} mrr={fmt(r['mrr'])} "
              f"cp={fmt(r['context_precision'])} cr={fmt(r['context_recall'])} "
              f"ctx_chars={fmt(r['mean_context_chars'], 0)} ms={fmt(r['mean_retrieval_ms'], 0)} "
              f"misses={results[-1]['misses']}")

    base = next(x for x in results if x["name"] == "baseline")
    lines = [
        f"| Experiment | Change | Chunks | Hit@1 | Hit@{args.k} | MRR | Δ MRR | Ctx precision | Ctx recall "
        f"| Avg context chars | ms/query | Missed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in results:
        change = (f"{x['mode']}" + (" + rerank" if x["rerank"] else "")
                  + (f", {x['model'].split('/')[-1]}" if x["model"] else "") + f", chunk {x['chunk_size']}")
        lines.append(
            f"| {x['name']} | {change} | {x['n_chunks']} | {fmt(x['hit_at_1'])} | {fmt(x['hit_rate'])} "
            f"| {fmt(x['mrr'])} | {x['mrr'] - base['mrr']:+.3f} | {fmt(x['context_precision'])} "
            f"| {fmt(x['context_recall'])} | {fmt(x['mean_context_chars'], 0)} | {fmt(x['mean_retrieval_ms'], 0)} "
            f"| {', '.join(x['misses']) or '-'} |")
    table = "\n".join(lines)
    out = PROJECT_ROOT / "data" / "eval"
    (out / "phase6_retrieval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "phase6_retrieval.md").write_text(
        "# Phase 6 retrieval experiments (30 answerable dev questions, k=%d)\n\n%s\n" % (args.k, table),
        encoding="utf-8")
    print("\n===== RESULTS =====\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
