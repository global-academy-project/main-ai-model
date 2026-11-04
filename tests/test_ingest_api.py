from fastapi.testclient import TestClient

from app import ingest_main


def _configure_environment(monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", "ingest-token")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-write-key")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("EMBEDDING_URL", "http://embedding:60110/v1/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")


def _request_body():
    return {
        "collection": "kb_guides",
        "chunks": [
            {
                "point_id": "00000000-0000-0000-0000-000000000001",
                "text": "작업 전 위험요인을 확인합니다.",
                "payload": {
                    "kb_source_id": 1,
                    "kb_chunk_id": 2,
                    "source_type": "KOSHA_GUIDE",
                    "site_id": 0,
                    "flagged": False,
                    "title": "안전 가이드",
                },
            }
        ],
    }


def test_ingest_chunks_authenticates_and_indexes(monkeypatch):
    _configure_environment(monkeypatch)
    calls = []

    def fake_index_chunks(settings, collection, chunks, *, write_api_key):
        calls.append((settings, collection, chunks, write_api_key))
        return len(chunks)

    monkeypatch.setattr(ingest_main, "index_chunks", fake_index_chunks)
    client = TestClient(ingest_main.app)
    response = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=_request_body(),
    )

    assert response.status_code == 200
    assert response.json() == {"indexed": 1}
    settings, collection, chunks, write_api_key = calls[0]
    assert settings.embedding_url == "http://embedding:60110/v1/embeddings"
    assert collection == "kb_guides"
    assert chunks[0].point_id == "00000000-0000-0000-0000-000000000001"
    assert chunks[0].payload["flagged"] is False
    assert write_api_key == "qdrant-write-key"


def test_ingest_chunks_rejects_missing_token(monkeypatch):
    _configure_environment(monkeypatch)
    client = TestClient(ingest_main.app)

    response = client.post("/ingest/chunks", json=_request_body())

    assert response.status_code == 401


def test_ingest_chunks_rejects_invalid_qdrant_point_id(monkeypatch):
    _configure_environment(monkeypatch)
    body = _request_body()
    body["chunks"][0]["point_id"] = "guide-chunk-1"
    client = TestClient(ingest_main.app)

    response = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 422