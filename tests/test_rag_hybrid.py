import json
from io import BytesIO

import pytest

from app.embedding import http
from app.embedding.config import RagSettings
from app.embedding.types import HybridEmbedding, SparseVector
from app.rag import indexing
from app.rag.qdrant import make_point, query_points, upsert_points


def test_qdrant_hybrid_query_and_write_use_separate_keys(monkeypatch):
    settings = RagSettings(
        embedding_url="http://embedding/v1/embeddings",
        qdrant_url="http://qdrant:6333",
        embedding_model="BAAI/bge-m3",
        embedding_dim=2,
        timeout_seconds=5,
        embedding_api_key=None,
        qdrant_api_key="reader-key",
    )
    embedding = HybridEmbedding(
        dense=[0.6, 0.8],
        sparse=SparseVector(indices=[4, 9], values=[1.0, 2.0]),
    )
    calls = []

    def respond(request, timeout):
        calls.append(request)
        result = {"status": "ok", "result": {"points": []}}
        return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(http, "urlopen", respond)

    assert query_points(settings, "kb_guides", embedding, [{"key": "flagged", "match": {"value": False}}], 5) == []
    query = json.loads(calls[0].data)
    assert calls[0].method == "POST"
    assert calls[0].headers["Api-key"] == "reader-key"
    assert query["query"] == {"fusion": "rrf"}
    assert [item["using"] for item in query["prefetch"]] == ["dense", "sparse"]
    assert query["prefetch"][1]["query"] == {"indices": [4, 9], "values": [1.0, 2.0]}

    point = make_point("point-id", embedding, {"text": "guide chunk"})
    upsert_points(settings, "kb_guides", [point], write_api_key="writer-key")
    upsert = json.loads(calls[1].data)
    assert calls[1].method == "PUT"
    assert calls[1].headers["Api-key"] == "writer-key"
    assert upsert["points"][0]["vector"]["dense"] == [0.6, 0.8]
    assert upsert["points"][0]["vector"]["sparse"] == {
        "indices": [4, 9], "values": [1.0, 2.0],
    }


def test_qdrant_upsert_requires_a_write_key():
    settings = RagSettings(
        embedding_url="http://embedding/v1/embeddings",
        qdrant_url="http://qdrant:6333",
        embedding_model="BAAI/bge-m3",
        embedding_dim=2,
        timeout_seconds=5,
        embedding_api_key=None,
        qdrant_api_key="reader-key",
    )
    with pytest.raises(ValueError, match="쓰기 키"):
        upsert_points(settings, "kb_guides", [{"id": "point-id"}], write_api_key=" ")


def test_index_chunks_batches_embedding_and_qdrant_upsert(monkeypatch):
    settings = RagSettings(
        embedding_url="http://embedding/v1/embeddings",
        qdrant_url="http://qdrant:6333",
        embedding_model="BAAI/bge-m3",
        embedding_dim=2,
        timeout_seconds=5,
        embedding_api_key=None,
        qdrant_api_key=None,
    )
    calls = []
    embedding = HybridEmbedding(
        dense=[0.6, 0.8], sparse=SparseVector(indices=[4], values=[1.0]),
    )
    monkeypatch.setattr(indexing, "embed_texts", lambda settings, texts: [embedding] * len(texts))
    monkeypatch.setattr(
        indexing,
        "upsert_points",
        lambda settings, collection, points, *, write_api_key: calls.append(
            (collection, points, write_api_key)
        ),
    )

    chunks = [
        indexing.IndexableChunk(
            point_id=f"point-{index}", text=f"text {index}", payload={"kb_chunk_id": index},
        )
        for index in range(3)
    ]
    assert indexing.index_chunks(
        settings, "kb_guides", chunks, write_api_key="writer-key", batch_size=2,
    ) == 3
    assert [len(call[1]) for call in calls] == [2, 1]
    assert all(call[0] == "kb_guides" and call[2] == "writer-key" for call in calls)