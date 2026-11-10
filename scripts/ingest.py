"""Phase 2: load raw documents, clean them, chunk them, and save chunks.jsonl.

Run from the project root (after scripts/download_docs.py):
    python scripts/ingest.py
    python scripts/ingest.py --chunk-size 500 --chunk-overlap 75   # experiment

Writes:
    data/processed/chunks.jsonl       one JSON chunk per line
    data/processed/ingest_stats.json  summary numbers (used in the README later)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter

from docuask.config import PROJECT_ROOT, get_settings
from docuask.ingest import chunk_units, load_corpus, save_chunks

EMBED_TOKEN_LIMIT = 256   # all-MiniLM-L6-v2 truncates input after 256 word pieces


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, default=s.chunking.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=s.chunking.chunk_overlap)
    args = p.parse_args()

    units, missing = load_corpus(s.paths.raw_dir, PROJECT_ROOT / "data" / "sources.csv")
    if missing:
        print(f"WARNING: {len(missing)} sources not downloaded: {', '.join(missing)}")
    if not units:
        print("No documents loaded. Run scripts/download_docs.py first.")
        return 1

    chunks = chunk_units(units, args.chunk_size, args.chunk_overlap)
    out = s.paths.processed_dir / "chunks.jsonl"
    save_chunks(chunks, out)

    lengths = [len(c.text) for c in chunks]
    est_tokens = [n / 4 for n in lengths]        # rough rule: ~4 characters per token in English
    docs = {u.doc_id for u in units}
    per_doc = Counter(c.doc_id for c in chunks)
    stats = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "documents": len(docs),
        "pdf_documents": len({u.doc_id for u in units if u.page is not None}),
        "html_documents": len({u.doc_id for u in units if u.page is None}),
        "units": len(units),
        "total_characters": sum(len(u.text) for u in units),
        "chunks": len(chunks),
        "chunk_chars_mean": round(statistics.mean(lengths), 1),
        "chunk_chars_median": statistics.median(lengths),
        "chunk_chars_min": min(lengths),
        "chunk_chars_max": max(lengths),
        "chunks_over_embed_limit_est": sum(t > EMBED_TOKEN_LIMIT for t in est_tokens),
        "chunks_per_doc_max": per_doc.most_common(1)[0],
    }
    (s.paths.processed_dir / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("===== INGEST STATS =====")
    for k, v in stats.items():
        print(f"  {k:30s} {v}")
    print("\nChunks per document:")
    for doc_id, n in sorted(per_doc.items()):
        print(f"  {doc_id:28s} {n}")
    sample = chunks[len(chunks) // 2]
    print(f"\nSample chunk {sample.chunk_id} {sample.citation}:\n{sample.text}\n")
    print(f"Saved {len(chunks)} chunks to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
