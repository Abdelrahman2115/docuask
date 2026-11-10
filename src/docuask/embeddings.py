"""Sentence-transformers wrapper that returns L2-normalized float32 vectors.

Some embedding models are trained with different prefixes for queries and for
passages (E5: "query: " / "passage: "; BGE: an instruction for queries only).
Using the wrong prefix quietly lowers retrieval quality, so the wrapper adds
them automatically based on the model name.
"""

from __future__ import annotations

import numpy as np

# model name -> (query prefix, passage prefix)
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/e5-small-v2": ("query: ", "passage: "),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.passage_prefix = MODEL_PREFIXES.get(model_name, ("", ""))

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # unit length -> dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]
