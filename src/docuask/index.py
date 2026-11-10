"""Build, open and search the Chroma vector index.

Design notes
- We compute embeddings ourselves and hand Chroma plain vectors. Chroma is only
  storage + nearest-neighbour search, so swapping the embedding model (Phase 6)
  never depends on Chroma's built-in embedding functions.
- Each embedding model gets its own collection ("docuask-all-minilm-l6-v2"), so
  English and multilingual indexes can live side by side for comparison.
- Rebuilding deletes and recreates the collection, so running the build twice
  never creates duplicates.
- Scores are cosine similarities computed from the returned vectors, so they do
  not depend on which distance metric the collection was created with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from docuask.ingest import Chunk


class EmbedderLike(Protocol):
    model_name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class SearchHit:
    chunk: Chunk
    score: float              # cosine similarity in [-1, 1]; higher is more similar


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def collection_name(base: str, model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return f"{base}-{slug}"[:63].rstrip("-")


def passage_text(chunk: Chunk) -> str:
    """Text that gets embedded: the document title on top of the chunk.

    A chunk such as 'Turn Bluetooth off when not in use...' doesn't say which
    guide it belongs to; adding the title gives the vector that context.
    The LLM still sees only chunk.text.
    """
    return f"{chunk.title}\n{chunk.text}"


def to_metadata(c: Chunk) -> dict:
    # Chroma metadata values must be str/int/float/bool, never None.
    return {
        "doc_id": c.doc_id, "title": c.title, "url": c.url, "source_file": c.source_file,
        "page": c.page if c.page is not None else -1,
        "section": c.section or "",
        "chunk_index": c.chunk_index,
    }


def from_metadata(chunk_id: str, text: str, m: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=m["doc_id"], title=m["title"], url=m["url"],
        source_file=m["source_file"],
        page=None if m["page"] == -1 else int(m["page"]),
        section=m["section"] or None,
        chunk_index=int(m["chunk_index"]), text=text,
    )


def get_client(index_dir: Path):
    import chromadb
    from chromadb.config import Settings

    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


# --------------------------------------------------------------------------- #
# Build / open / search
# --------------------------------------------------------------------------- #
def build_index(chunks: list[Chunk], embedder: EmbedderLike, index_dir: Path, name: str,
                batch_size: int = 128):
    """(Re)create collection `name` from scratch and return it."""
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("chunk_ids are not unique; re-run scripts/ingest.py")
    client = get_client(index_dir)
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 - collection did not exist yet
        pass
    col = client.create_collection(name=name, metadata={"embedding_model": embedder.model_name})

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vecs = embedder.embed_passages([passage_text(c) for c in batch])
        col.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vecs.tolist(),
            documents=[c.text for c in batch],
            metadatas=[to_metadata(c) for c in batch],
        )
    return col


def open_collection(index_dir: Path, name: str):
    client = get_client(index_dir)
    try:
        return client.get_collection(name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Index '{name}' not found in {index_dir}. Run: python scripts/build_index.py"
        ) from e


def search(collection, embedder: EmbedderLike, query: str, k: int = 4) -> list[SearchHit]:
    q = embedder.embed_query(query)
    res = collection.query(
        query_embeddings=[q.tolist()],
        n_results=k,
        include=["documents", "metadatas", "embeddings"],
    )
    hits = []
    for cid, doc, meta, emb in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
        score = float(np.dot(q, np.asarray(emb, dtype=np.float32)))
        hits.append(SearchHit(from_metadata(cid, doc, meta), score))
    return sorted(hits, key=lambda h: h.score, reverse=True)
