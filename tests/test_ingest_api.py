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
                    "text": "작업 전 위험요인을 확인합니다.",
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

    monkeypatch.setattr(ingest_main, "validate_chunks", lambda settings, collection, chunks: None)
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


def test_save_local_vectors_embeds_in_batches_without_qdrant(monkeypatch, tmp_path):
    from app.rag.embedding import HybridEmbedding, SparseVector

    _configure_environment(monkeypatch)
    monkeypatch.delenv("QDRANT_API_KEY")
    monkeypatch.setenv("LOCAL_VECTOR_DIR", str(tmp_path / "vectors"))
    body = _request_body()
    body["dataset_id"] = "law-v1"
    body["chunks"] = []
    for index in range(9):
        chunk = deepcopy(_request_body()["chunks"][0])
        chunk["point_id"] = str(uuid5(NAMESPACE_URL, f"LAW:local:{index}"))
        chunk["text"] = f"법령 청크 {index}"
        chunk["payload"]["kb_chunk_id"] = index + 1
        chunk["payload"]["text"] = chunk["text"]
        body["chunks"].append(chunk)

    batch_calls = []

    def fake_embed_texts(settings, texts):
        batch_calls.append(texts)
        return [
            HybridEmbedding([0.6, 0.8], SparseVector([4], [1.0]))
            for _ in texts
        ]

    monkeypatch.setattr(ingest_main, "embed_texts", fake_embed_texts)
    response = TestClient(ingest_main.app).post(
        "/ingest/vectors/local",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 200
    assert response.json() == {
        "saved": 9,
        "dataset_id": "law-v1",
        "file": "law-v1.jsonl",
        "model": "BAAI/bge-m3",
    }
    assert [len(batch) for batch in batch_calls] == [8, 1]
    saved = [
        json.loads(line)
        for line in (tmp_path / "vectors" / "law-v1.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(saved) == 9
    assert saved[0]["collection"] == "kb_guides"
    assert saved[0]["model"] == "BAAI/bge-m3"
    assert saved[0]["vector"] == {
        "dense": [0.6, 0.8],
        "sparse": {"indices": [4], "values": [1.0]},
    }
    assert saved[0]["payload"]["text"] == "법령 청크 0"


def test_save_local_vectors_requires_auth_and_safe_dataset_id(monkeypatch, tmp_path):
    _configure_environment(monkeypatch)
    monkeypatch.setenv("LOCAL_VECTOR_DIR", str(tmp_path))
    body = _request_body()
    body["dataset_id"] = "../outside"
    client = TestClient(ingest_main.app)

    unauthorized = client.post("/ingest/vectors/local", json=body)
    invalid_id = client.post(
        "/ingest/vectors/local",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert unauthorized.status_code == 401
    assert invalid_id.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_save_local_vectors_failure_keeps_previous_file(monkeypatch, tmp_path):
    from app.rag.errors import RagBackendError

    _configure_environment(monkeypatch)
    output_dir = tmp_path / "vectors"
    output_dir.mkdir()
    monkeypatch.setenv("LOCAL_VECTOR_DIR", str(output_dir))
    target = output_dir / "law-v1.jsonl"
    target.write_text("previous data\n", encoding="utf-8")

    def fail_embedding(settings, texts):
        raise RagBackendError("embedding unavailable")

    monkeypatch.setattr(ingest_main, "embed_texts", fail_embedding)
    body = _request_body()
    body["dataset_id"] = "law-v1"
    response = TestClient(ingest_main.app).post(
        "/ingest/vectors/local",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "임베딩 요청이 실패했습니다."}
    assert target.read_text(encoding="utf-8") == "previous data\n"
    assert sorted(path.name for path in output_dir.iterdir()) == ["law-v1.jsonl"]


def test_save_local_chunk_file_vectors_reads_validated_chunks(monkeypatch, tmp_path):
    from app.rag.embedding import HybridEmbedding, SparseVector

    _configure_environment(monkeypatch)
    chunk_dir = tmp_path / "law"
    chunk_dir.mkdir()
    output_dir = tmp_path / "vectors"
    monkeypatch.setenv("LOCAL_CHUNK_DIR", str(chunk_dir))
    monkeypatch.setenv("LOCAL_VECTOR_DIR", str(output_dir))

    records = []
    for index in range(9):
        records.append({
            "natural_key": {
                "law_key": "산업안전보건기준에관한규칙",
                "article_key": str(index + 1),
                "part": 0,
            },
            "text": f"제{index + 1}조 원문",
            "embed_text": f"법령 머리말\n제{index + 1}조 원문",
            "collection": "kb_guides",
            "kb_source_id": None,
            "kb_chunk_id": None,
            "point_id": None,
        })
    (chunk_dir / "05_chunks.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (chunk_dir / "06_validation.json").write_text(
        json.dumps({"ok": True, "summary": {"chunks": len(records)}}),
        encoding="utf-8",
    )
    (chunk_dir / "chunk_index_map.json").write_text(
        json.dumps({
            "source_type": "LAW",
            "external_id": "산업안전보건기준에관한규칙",
            "indexes": {
                f"산업안전보건기준에관한규칙|{index + 1}|0": index
                for index in range(len(records))
            },
        }),
        encoding="utf-8",
    )

    batch_calls = []

    def fake_embed_texts(settings, texts):
        batch_calls.append(texts)
        return [
            HybridEmbedding([0.6, 0.8], SparseVector([4], [1.0]))
            for _ in texts
        ]

    monkeypatch.setattr(ingest_main, "embed_texts", fake_embed_texts)
    response = TestClient(ingest_main.app).post(
        "/ingest/vectors/local/file",
        headers={"Authorization": "Bearer ingest-token"},
        json={"dataset_id": "law-local", "chunk_file": "05_chunks.jsonl"},
    )

    assert response.status_code == 200
    assert response.json()["saved"] == 9
    assert response.json()["file"] == "law-local.jsonl"
    assert [len(batch) for batch in batch_calls] == [8, 1]
    saved = [
        json.loads(line)
        for line in (output_dir / "law-local.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(saved) == 9
    assert set(saved[0]) == {"id", "vector", "payload"}
    assert saved[0]["id"] == str(uuid5(NAMESPACE_URL, "LAW:산업안전보건기준에관한규칙:0"))
    assert saved[0]["vector"]["dense"] == [0.6, 0.8]
    assert saved[0]["vector"]["sparse"] == {"indices": [4], "values": [1.0]}
    assert saved[0]["payload"]["text"] == records[0]["text"]
    assert saved[0]["payload"]["kb_source_id"] is None


def test_save_local_chunk_file_requires_passing_validation(monkeypatch, tmp_path):
    _configure_environment(monkeypatch)
    chunk_dir = tmp_path / "law"
    chunk_dir.mkdir()
    monkeypatch.setenv("LOCAL_CHUNK_DIR", str(chunk_dir))
    (chunk_dir / "05_chunks.jsonl").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "06_validation.json").write_text(
        json.dumps({"ok": False, "summary": {"chunks": 1}}),
        encoding="utf-8",
    )

    response = TestClient(ingest_main.app).post(
        "/ingest/vectors/local/file",
        headers={"Authorization": "Bearer ingest-token"},
        json={"dataset_id": "law-local", "chunk_file": "05_chunks.jsonl"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "청크 검증을 통과하지 못했습니다."}


def test_ingest_finalizes_sources_after_ledger_check_and_index(monkeypatch):
    _configure_environment(monkeypatch)
    body = _request_body()
    body["finalize_sources"] = [{"source_id": 1, "chunk_count": 1}]
    calls = []

    monkeypatch.setattr(
        ingest_main,
        "validate_chunks",
        lambda settings, collection, chunks: calls.append("validate"),
    )
    monkeypatch.setattr(
        ingest_main,
        "index_chunks",
        lambda settings, collection, chunks, *, write_api_key: calls.append("index") or len(chunks),
    )
    monkeypatch.setattr(
        ingest_main,
        "mark_sources_embedded",
        lambda settings, expected_chunk_counts, embedding_model: calls.append(
            ("embedded", expected_chunk_counts, embedding_model)
        ),
    )

    client = TestClient(ingest_main.app)
    response = client.post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 200
    assert calls == ["validate", "index", ("embedded", {1: 1}, "BAAI/bge-m3")]


def test_ingest_rejects_finalization_for_source_not_in_batch(monkeypatch):
    _configure_environment(monkeypatch)
    body = _request_body()
    body["finalize_sources"] = [{"source_id": 2, "chunk_count": 1}]

    response = TestClient(ingest_main.app).post(
        "/ingest/chunks",
        headers={"Authorization": "Bearer ingest-token"},
        json=body,
    )

    assert response.status_code == 422


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
    from app.rag.embedding import HybridEmbedding, SparseVector

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
    monkeypatch.setattr(
        indexing,
        "embed_texts",
        lambda settings, texts: [
            HybridEmbedding([0.6, 0.8], SparseVector([], [])) for _ in texts
        ],
    )
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

    def fake_post_json(url, body, headers, timeout, service_name):
        calls.append((url, body, headers, timeout, service_name))
        return {"status": "ok"}

    monkeypatch.setattr(qdrant, "post_json", fake_post_json)
    point_ids = ["00000000-0000-0000-0000-000000000001", 7]

    qdrant.delete_points(settings, "kb_accidents", point_ids, write_api_key="writer")

    url, body, headers, timeout, service_name = calls[0]
    assert url == "http://qdrant:6333/collections/kb_accidents/points/delete?wait=true"
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


def test_catalog_validates_chunk_identity_and_marks_expected_count(monkeypatch):
    from types import SimpleNamespace

    from app.rag import catalog
    from app.rag.config import RagSettings
    from app.rag.errors import RagBackendError
    from app.rag.indexing import IndexableChunk

    point_id = "00000000-0000-5000-8000-000000000001"
    executed = []

    class Cursor:
        query = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query, params):
            self.query = query
            executed.append((query, params))

        def fetchall(self):
            if "LEFT JOIN kb_chunks" in self.query:
                return [(1, 2)]
            return [(2, 1, "kb_guides", point_id, "법령 조문", None, "LAW", None, False)]

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

        def rollback(self):
            pass

    class MySQLError(Exception):
        pass

    connection = Connection()
    driver = SimpleNamespace(
        MySQLError=MySQLError,
        connect=lambda **kwargs: connection,
    )
    monkeypatch.setattr(catalog, "_mysql_driver", lambda: driver)
    settings = RagSettings(
        embedding_url="http://embedding",
        qdrant_url="http://qdrant:6333",
        embedding_model="BAAI/bge-m3",
        embedding_dim=2,
        timeout_seconds=5,
        embedding_api_key=None,
        qdrant_api_key=None,
        kb_db_host="mysql",
        kb_ingest_password="secret",
    )
    chunk = IndexableChunk(
        point_id=point_id,
        text="법령 조문",
        payload={
            "kb_source_id": 1,
            "kb_chunk_id": 2,
            "source_type": "LAW",
            "site_id": 0,
            "flagged": False,
            "text": "법령 조문",
        },
    )

    catalog.validate_chunks(settings, "kb_guides", [chunk])
    with pytest.raises(RagBackendError, match="청크 수"):
        catalog.mark_sources_embedded(settings, {1: 3}, "BAAI/bge-m3")

    assert executed[0][1] == [2]
    assert executed[1][1] == [1]
    assert connection is not None