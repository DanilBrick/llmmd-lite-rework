from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from qdrant_client.models import SparseVector


@dataclass
class HybridEncoder:
    dense_model: Any
    sparse_model: Any | None
    device: str

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed: list[str] = []
        for t in texts:
            s = (t or "").strip()
            low = s[:12].lower()
            prefixed.append(s if low.startswith("passage:") else f"passage: {s}")
        return np.asarray(
            self.dense_model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 32,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        q = text.strip()
        if not q.lower().startswith("query:"):
            q = f"query: {q}"
        v = self.dense_model.encode(
            [q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)[0]

    def encode_sparse(self, texts: list[str]) -> list[SparseVector] | None:
        if self.sparse_model is None:
            return None
        out: list[SparseVector] = []
        for emb in self.sparse_model.embed(texts):
            idx = getattr(emb, "indices", None)
            val = getattr(emb, "values", None)
            if idx is None or val is None:
                continue
            if hasattr(idx, "tolist"):
                idx = idx.tolist()
            if hasattr(val, "tolist"):
                val = val.tolist()
            out.append(SparseVector(indices=list(idx), values=list(val)))
        return out if len(out) == len(texts) else None


def load_hybrid_encoder(
    model_name: str,
    *,
    device: str | None,
    enable_hybrid: bool,
) -> tuple[HybridEncoder, bool]:
    import torch
    from sentence_transformers import SentenceTransformer

    if device:
        dev = device
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        dense = SentenceTransformer(model_name, device=dev)
    except Exception:
        dense = SentenceTransformer(model_name, device="cpu")
        dev = "cpu"

    sparse = None
    hybrid_on = False
    if enable_hybrid:
        try:
            from fastembed import SparseTextEmbedding

            sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
            hybrid_on = True
        except Exception:
            sparse = None
            hybrid_on = False

    return HybridEncoder(dense_model=dense, sparse_model=sparse, device=dev), hybrid_on
