import json
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

from fastapi.testclient import TestClient
import pytest

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


def _near_miss_request_body():
    near_miss_id = 12
    text = "배관 랙 발판 미끄러짐. 검토 태그: NEAR_MISS, FALL, HEIGHT."
    return {
        "collection": "kb_accidents",
        "chunks": [
            {
                "point_id": str(uuid5(NAMESPACE_URL, f"NEAR_MISS:{near_miss_id}:0")),
                "text": text,
                "payload": {
                    "kb_source_id": near_miss_id,
                    "kb_chunk_id": near_miss_id,
                    "source_type": "NEAR_MISS",
                    "site_id": 3,
                    "flagged": False,
                    "severity": "NEAR_MISS",
                    "accident_type": ["FALL"],
                    "hazard": ["HUMAN"],
                    "work_type": ["HEIGHT"],
                    "cas": [],
                    "ghs": [],
                    "year": 2026,
                    "occurred_at": "2026-11-03T10:20:00",
                    "zone_id": 5,
                    "title": "배관 랙 발판 미끄러짐",
                    "text": text,
                },
            }
        ],
    }


def test_ingest_chunks_authenticates_and_indexes(monkeypatch):
    _configure_environment(monkeypatch)
    monkeypatch.setenv("KB_INGEST_API_TOKEN", "ingest-token\r")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-write-key\r")
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


def test_ingest_near_miss_request_contract(monkeypatch):
    _configure_environment(monkeypatch)
    calls = []

    def fake_index_chunks(settings, collection, chunks, *, write_api_key):
        calls.append((collection, chunks))
        return len(chunks)

    monkeypatch.setattr(ingest_main, "index_chunks", fake_index_chunks)
    client = TestClient(ingest_main.app)
    body = _near_miss_request_body()

    response = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 200
    assert response.json() == {"indexed": 1}
    collection, chunks = calls[0]
    assert collection == "kb_accidents"
    assert chunks[0].point_id == body["chunks"][0]["point_id"]
    assert chunks[0].payload == body["chunks"][0]["payload"]
    assert chunks[0].text == chunks[0].payload["text"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda chunk: chunk["payload"].update(flagged=True),
        lambda chunk: chunk["payload"].update(kb_chunk_id=13),
        lambda chunk: chunk.update(point_id="00000000-0000-0000-0000-000000000001"),
        lambda chunk: chunk["payload"].update(year=2025),
        lambda chunk: chunk["payload"].update(year=True),
        lambda chunk: chunk["payload"].update(ghs=["H330"]),
        lambda chunk: chunk["payload"].update(text="payload와 다른 원문"),
        lambda chunk: chunk["payload"].update(accident_type="FALL"),
    ],
)
def test_ingest_near_miss_rejects_contract_mismatches(monkeypatch, mutate):
    _configure_environment(monkeypatch)
    body = _near_miss_request_body()
    mutate(body["chunks"][0])
    client = TestClient(ingest_main.app)

    response = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 422


def test_delete_points_authenticates_and_deletes(monkeypatch):
    _configure_environment(monkeypatch)
    calls = []

    def fake_delete_points(settings, collection, point_ids, *, write_api_key):
        calls.append((settings, collection, point_ids, write_api_key))

    monkeypatch.setattr(ingest_main, "delete_qdrant_points", fake_delete_points)
    client = TestClient(ingest_main.app)
    point_ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]

    response = client.post(
        "/ingest/points/delete",
        headers={"Authorization": "Bearer ingest-token"},
        json={"collection": "kb_accidents", "point_ids": point_ids},
    )

    assert response.status_code == 200
    assert response.json() == {"deleted": 2}
    settings, collection, deleted_ids, write_api_key = calls[0]
    assert settings.qdrant_url == "http://qdrant:6333"
    assert collection == "kb_accidents"
    assert deleted_ids == point_ids
    assert write_api_key == "qdrant-write-key"


def test_delete_points_requires_authentication(monkeypatch):
    _configure_environment(monkeypatch)
    client = TestClient(ingest_main.app)

    response = client.post(
        "/ingest/points/delete",
        json={"collection": "kb_accidents", "point_ids": [1]},
    )

    assert response.status_code == 401


def test_delete_points_rejects_invalid_or_unbounded_requests(monkeypatch):
    _configure_environment(monkeypatch)
    client = TestClient(ingest_main.app)
    headers = {"Authorization": "Bearer ingest-token"}

    invalid_id = client.post(
        "/ingest/points/delete",
        headers=headers,
        json={"collection": "kb_accidents", "point_ids": ["not-a-uuid"]},
    )
    too_many_ids = client.post(
        "/ingest/points/delete",
        headers=headers,
        json={"collection": "kb_accidents", "point_ids": list(range(33))},
    )

    assert invalid_id.status_code == 422
    assert too_many_ids.status_code == 422


def test_rag_settings_strips_secret_whitespace(monkeypatch):
    _configure_environment(monkeypatch)
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-key\r")
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "reader-key\r")

    from app.rag.config import RagSettings

    settings = RagSettings.from_env()

    assert settings.embedding_api_key == "embedding-key"
    assert settings.qdrant_api_key == "reader-key"


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


def test_index_chunks_embeds_and_upserts_dense_points(monkeypatch):
    from app.rag import indexing
    from app.rag.config import RagSettings

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
    monkeypatch.setattr(indexing, "embed_text", lambda settings, text: [0.6, 0.8])
    monkeypatch.setattr(
        indexing,
        "upsert_points",
        lambda settings, collection, points, *, write_api_key: calls.append(
            (collection, points, write_api_key)
        ),
    )

    chunk = indexing.IndexableChunk(
        point_id="00000000-0000-0000-0000-000000000001",
        text="법령 조문 테스트",
        payload={"source_type": "LAW", "site_id": 0},
    )

    assert indexing.index_chunks(settings, "kb_guides", [chunk], write_api_key="writer") == 1
    assert calls[0][0] == "kb_guides"
    assert calls[0][1][0]["vector"] == {"dense": [0.6, 0.8]}
    assert calls[0][2] == "writer"


def test_delete_points_sends_qdrant_points_selector(monkeypatch):
    from app.rag import qdrant
    from app.rag.config import RagSettings

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

    def fake_delete_json(url, body, headers, timeout, service_name):
        calls.append((url, body, headers, timeout, service_name))
        return {"status": "ok"}

    monkeypatch.setattr(qdrant, "delete_json", fake_delete_json)
    point_ids = ["00000000-0000-0000-0000-000000000001", 7]

    qdrant.delete_points(settings, "kb_accidents", point_ids, write_api_key="writer")

    url, body, headers, timeout, service_name = calls[0]
    assert url == "http://qdrant:6333/collections/kb_accidents/points?wait=true"
    assert body == {"points": point_ids}
    assert headers["api-key"] == "writer"
    assert timeout == 5
    assert service_name == "Qdrant 삭제"


def test_delete_json_uses_http_delete_with_points_body(monkeypatch):
    from app.rag import http

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"status":"ok"}'

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(http, "urlopen", fake_urlopen)

    response = http.delete_json(
        "http://qdrant:6333/collections/kb_accidents/points?wait=true",
        {"points": [1]},
        {"api-key": "writer"},
        5,
        "Qdrant 삭제",
    )

    request, timeout = calls[0]
    assert response == {"status": "ok"}
    assert request.get_method() == "DELETE"
    assert json.loads(request.data) == {"points": [1]}
    assert request.get_header("Api-key") == "writer"
    assert timeout == 5