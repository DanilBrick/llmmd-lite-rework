from __future__ import annotations

import pytest

from rag_service.store import QdrantSchemaError, collection_schema_error, ensure_collection


def test_collection_schema_accepts_hybrid_named_vectors():
    info = {
        "config": {
            "params": {
                "vectors": {"dense": {"size": 1024}},
                "sparse_vectors": {"sparse": {}},
            }
        }
    }

    assert collection_schema_error(info, dense_size=1024, hybrid=True) is None


def test_collection_schema_rejects_dense_only_collection_for_hybrid_search():
    info = {
        "config": {
            "params": {
                "vectors": {"size": 1024},
            }
        }
    }

    error = collection_schema_error(info, dense_size=1024, hybrid=True)

    assert error is not None
    assert "named dense vector" in error


def test_ensure_collection_raises_on_incompatible_existing_schema():
    class Client:
        def get_collection(self, _name):
            return {"config": {"params": {"vectors": {"size": 384}}}}

    with pytest.raises(QdrantSchemaError) as exc:
        ensure_collection(Client(), "corpus", dense_size=1024, hybrid=False)

    assert "recreate_collection=true" in str(exc.value)


def test_ensure_collection_creates_missing_hybrid_collection():
    class Client:
        def __init__(self):
            self.created = None

        def get_collection(self, _name):
            raise RuntimeError("missing")

        def create_collection(self, **kwargs):
            self.created = kwargs

    client = Client()

    ensure_collection(client, "corpus", dense_size=1024, hybrid=True)

    assert client.created is not None
    assert "dense" in client.created["vectors_config"]
    assert "sparse" in client.created["sparse_vectors_config"]
