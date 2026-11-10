"""Search the index from the command line (retrieval only, no LLM).

    python scripts/search.py "how do I stop robocalls"
    python scripts/search.py "roaming charges abroad" --k 5
"""

from __future__ import annotations

import argparse
import time

from docuask.config import get_settings
from docuask.embeddings import Embedder
from docuask.index import collection_name, open_collection, search


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=s.retrieval.top_k)
    p.add_argument("--model", default=s.embeddings.model)
    args = p.parse_args()

    embedder = Embedder(args.model)
    col = open_collection(s.paths.index_dir, collection_name(s.vector_store.collection, args.model))
    t0 = time.perf_counter()
    hits = search(col, embedder, args.query, k=args.k)
    ms = 1000 * (time.perf_counter() - t0)
    print(f"Top {len(hits)} for: {args.query!r}  ({ms:.0f} ms)\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. score {h.score:.3f}  {h.chunk.citation}  {h.chunk.title}")
        print("   " + h.chunk.text[:300].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
