from __future__ import annotations

import uuid
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, Fusion, FusionQuery, PointStruct, Prefetch, SparseVector, VectorParams

from .embeddings import HybridEncoder


class QdrantSchemaError(RuntimeError):
    """Raised when an existing Qdrant collection does not match llmmd schema."""


def stable_point_id(rel_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"llmmd:{rel_path}:{chunk_index}"))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _collection_params(info: Any) -> Any:
    config = _field(info, "config")
    return _field(config, "params", config)


def _vector_size(vectors: Any, name: str | None = None) -> int | None:
    if isinstance(vectors, dict):
        item = vectors.get(name or "") if name else None
        if item is None and not name and "size" in vectors:
            item = vectors
    else:
        item = vectors
    size = _field(item, "size")
    return int(size) if size is not None else None


def collection_schema_error(info: Any, *, dense_size: int, hybrid: bool) -> str | None:
    params = _collection_params(info)
    vectors = _field(params, "vectors")
    sparse = _field(params, "sparse_vectors") or _field(params, "sparse_vectors_config")

    if hybrid:
        if not isinstance(vectors, dict) or "dense" not in vectors:
            return "expected named dense vector 'dense' for hybrid search"
        actual_dense = _vector_size(vectors, "dense")
        if actual_dense != dense_size:
            return f"expected dense vector size {dense_size}, got {actual_dense}"
        if not sparse or (isinstance(sparse, dict) and "sparse" not in sparse):
            return "expected sparse vector 'sparse' for hybrid search"
        return None

    if isinstance(vectors, dict) and "dense" in vectors:
        return "expected unnamed dense vector for dense-only search, got named 'dense'"
    actual_dense = _vector_size(vectors)
    if actual_dense != dense_size:
        return f"expected dense vector size {dense_size}, got {actual_dense}"
    return None


def ensure_collection(
    client: QdrantClient,
    name: str,
    dense_size: int,
    hybrid: bool,
) -> None:
    info: Any = None
    try:
        info = client.get_collection(name)
    except Exception:
        info = None

    if info is not None:
        mismatch = collection_schema_error(info, dense_size=dense_size, hybrid=hybrid)
        if mismatch:
            raise QdrantSchemaError(
                f"Collection '{name}' has incompatible schema: {mismatch}. "
                "Use POST /v1/index with recreate_collection=true or change collection_name."
            )
        return

    if hybrid:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dense_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": rest.SparseVectorParams()},
        )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dense_size, distance=Distance.COSINE),
        )


def recreate_collection(
    client: QdrantClient,
    name: str,
    dense_size: int,
    hybrid: bool,
) -> None:
    if hybrid:
        client.recreate_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dense_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": rest.SparseVectorParams()},
        )
    else:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dense_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    encoder: HybridEncoder,
    hybrid: bool,
    *,
    corpus_root: Path,
    rel_path: str,
    chunks: list[tuple[str, str, int]],
    batch_size: int = 64,
) -> int:
    """
    chunks: (text, heading, chunk_index)
    """
    corpus_root = corpus_root.resolve()
    written = 0
    indexed_at_unix = time.time()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [t for t, _, _ in batch]
        dense = encoder.encode_passages(texts)
        sparse_list: list[SparseVector] | None = None
        if hybrid and encoder.sparse_model is not None:
            sparse_list = encoder.encode_sparse(texts)

        points: list[PointStruct] = []
        for row_idx, (text, heading, chunk_index) in enumerate(batch):
            pid = stable_point_id(rel_path, chunk_index)
            rel = Path(rel_path)
            payload = {
                "text": text,
                "heading": heading,
                "chunk_index": chunk_index,
                "source_path": rel_path.replace("\\", "/"),
                "file_name": rel.name,
                "parent_dir": rel.parent.as_posix() if rel.parent != Path(".") else "",
                "corpus_root": str(corpus_root),
                "indexed_at_unix": indexed_at_unix,
                "chunk_text_chars": len(text),
            }
            if hybrid and sparse_list:
                vec: Any = {"dense": dense[row_idx].tolist(), "sparse": sparse_list[row_idx]}
            else:
                vec = dense[row_idx].tolist()
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
        client.upsert(collection_name=collection, points=points)
        written += len(points)
    return written


def search_dense(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    limit: int,
    score_threshold: float | None = None,
    *,
    named_dense: bool = False,
) -> list[rest.ScoredPoint]:
    qv: Any = ("dense", query_vector.tolist()) if named_dense else query_vector.tolist()
    return client.search(
        collection_name=collection,
        query_vector=qv,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )


def search_hybrid(
    client: QdrantClient,
    collection: str,
    dense_query: np.ndarray,
    sparse_query: SparseVector,
    limit: int,
) -> list[rest.ScoredPoint]:
    res = client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=sparse_query, using="sparse", limit=max(limit * 4, 20)),
            Prefetch(query=dense_query.tolist(), using="dense", limit=max(limit * 4, 20)),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return list(res.points)
