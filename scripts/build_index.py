"""Phase 3: embed all chunks and (re)build the Chroma index, then run sanity queries.

Run from the project root (after scripts/ingest.py):
    python scripts/build_index.py
    python scripts/build_index.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    python scripts/build_index.py --chunk-size 400

Writes data/index/ (Chroma files) and data/index/index_meta_<collection>.json.
The first run downloads the embedding model (~90 MB) into the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

from docuask.config import get_settings
from docuask.embeddings import Embedder
from docuask.index import build_index, search
from docuask.variants import chunks_path as variant_chunks_path, index_collection
from docuask.ingest import load_chunks

# Quick smoke test, not the evaluation (that is Phase 5): each question names the
# guide that obviously answers it. If these fail, something is broken.
SANITY = [
    ("How do I keep my phone number when I switch carriers?", "porting"),
    ("What is cramming on a phone bill?", "telephone_bill"),
    ("Someone switched my long distance company without my permission", "slamming"),
    ("I got a missed call from an unknown international number that rang once", "one_ring_scam"),
    ("Can I send a text message to 911 in an emergency?", "text_to_911"),
]


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=s.embeddings.model)
    p.add_argument("--chunk-size", type=int, default=s.chunking.chunk_size)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--quiet", action="store_true", help="skip the sanity query printout")
    args = p.parse_args()

    chunks_path = variant_chunks_path(s, args.chunk_size)
    if not chunks_path.exists():
        print(f"{chunks_path.name} not found. Run: python scripts/ingest.py --chunk-size {args.chunk_size}")
        return 1
    chunks = load_chunks(chunks_path)
    name = index_collection(s, args.model, args.chunk_size)

    t0 = time.perf_counter()
    embedder = Embedder(args.model)
    t_load = time.perf_counter() - t0
    print(f"Model {args.model}: dim={embedder.dim}, max_seq_length={embedder.max_seq_length}, loaded in {t_load:.1f}s")

    t0 = time.perf_counter()
    col = build_index(chunks, embedder, s.paths.index_dir, name)
    t_build = time.perf_counter() - t0
    count = col.count()
    print(f"Indexed {count} chunks into collection '{name}' in {t_build:.1f}s "
          f"({1000 * t_build / max(count, 1):.0f} ms/chunk)")
    if count != len(chunks):
        print(f"ERROR: expected {len(chunks)} vectors, found {count}")
        return 1

    print("\n===== SANITY QUERIES =====")
    hits_ok, latencies = 0, []
    for question, expected in SANITY:
        t0 = time.perf_counter()
        hits = search(col, embedder, question, k=args.k)
        latencies.append(1000 * (time.perf_counter() - t0))
        found = any(h.chunk.doc_id == expected for h in hits)
        hits_ok += found
        if not args.quiet:
            print(f"\nQ: {question}\n   expected {expected}: {'HIT' if found else 'MISS'}")
            for h in hits:
                print(f"   {h.score:.3f}  {h.chunk.citation}  {h.chunk.text[:70].replace(chr(10), ' ')}...")
    print(f"\nSanity hits: {hits_ok}/{len(SANITY)} in top-{args.k}; "
          f"mean query latency {sum(latencies) / len(latencies):.0f} ms")

    meta = {
        "collection": name,
        "embedding_model": args.model,
        "dim": embedder.dim,
        "max_seq_length": embedder.max_seq_length,
        "chunks": count,
        "chunk_size": args.chunk_size,
        "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "model_load_seconds": round(t_load, 2),
        "index_build_seconds": round(t_build, 2),
        "sanity_hits": f"{hits_ok}/{len(SANITY)}",
        "mean_query_ms": round(sum(latencies) / len(latencies), 1),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path = s.paths.index_dir / f"index_meta_{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
