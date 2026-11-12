"""ingest_law_vectors: 벡터 파일 만들기 요청에 data_type·subdir를 싣는다 (A의 새 API: <LOCAL_CHUNK_ROOT>/<data_type>/<subdir>/).

서버·Qdrant 없이 가짜 세션과 임시 폴더로 돈다. 기존 시험과 겹치지 않게 별도 파일로 둔다.
"""

from __future__ import annotations

import argparse
import json

import pytest

from scripts import ingest_law_vectors as m

DIM = 3
LAW = "산업안전보건기준에관한규칙"


class _R:
    def __init__(self, data, status=200):
        self.status_code, self._data, self.text = status, data, json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


def record(n: int) -> dict:
    return {
        "natural_key": {"law_key": LAW, "article_key": f"제{n}조", "part": 0},
        "source_type": "LAW", "collection": "kb_guides", "site_id": 0, "flagged": False,
        "text": f"본문{n}", "embed_text": f"임베딩{n}", "locator": f"제{n}조", "title": "산업안전보건기준에 관한 규칙",
    }


def prepare(tmp_path, n=2):
    chunks = tmp_path / "05_chunks.jsonl"
    chunks.write_text("".join(json.dumps(record(i), ensure_ascii=False) + "\n" for i in range(n)), encoding="utf-8")
    (tmp_path / "06_validation.json").write_text(json.dumps({"ok": True, "summary": {"chunks": n}}), encoding="utf-8")
    m.save_index_map(tmp_path / "chunk_index_map.json", LAW, {m.natural_key_str(record(i)): i for i in range(n)})
    return chunks


def parse(chunks, *extra):
    return m.parse_args(["--chunks", str(chunks), "--dataset-id", "ds", "--embedding-dim", str(DIM), *extra])


# ── 옵션 ──────────────────────────────────────────────────────────────────────

def test_defaults_are_guides_and_law():
    args = m.parse_args(["--dry-run"])
    assert (args.data_type, args.subdir) == ("guides", "law")
    assert m.server_location(args) == ("guides", "law")


def test_custom_subdir_and_legacy_none():
    assert m.server_location(m.parse_args(["--subdir", "law-2026"])) == ("guides", "law-2026")
    assert m.server_location(m.parse_args(["--data-type", "none"])) == (None, None)


@pytest.mark.parametrize("extra", [
    ["--subdir", "../x"], ["--subdir", ""], ["--subdir", "a b"], ["--subdir", "x" * 101],
    ["--data-type", "accidents"], ["--data-type", "none", "--subdir", "other"],
])
def test_bad_options_are_rejected(extra):
    with pytest.raises(SystemExit):
        m.parse_args(extra)


def test_namespace_without_the_new_attributes_means_old_request():
    assert m.server_location(argparse.Namespace()) == (None, None)


# ── 요청 본문 ─────────────────────────────────────────────────────────────────

class Capture:
    def __init__(self, response):
        self.response, self.bodies = response, []

    def post(self, url, **kw):
        self.bodies.append(kw["json"])
        return self.response


def ok_response():
    return _R({"saved": 2, "dataset_id": "ds", "file": "ds.jsonl", "collection": "kb_guides", "model": "fake"})


def test_save_vectors_sends_data_type_and_subdir():
    session = Capture(ok_response())
    m.save_vectors(session, "http://127.0.0.1:8001", "t", chunk_file="05_chunks.jsonl", dataset_id="ds", count=2,
                   data_type="guides", subdir="law")
    assert session.bodies == [{"dataset_id": "ds", "chunk_file": "05_chunks.jsonl", "data_type": "guides", "subdir": "law"}]


def test_save_vectors_without_location_sends_the_old_body():
    session = Capture(ok_response())
    m.save_vectors(session, "http://127.0.0.1:8001", "t", chunk_file="05_chunks.jsonl", dataset_id="ds", count=2)
    assert session.bodies == [{"dataset_id": "ds", "chunk_file": "05_chunks.jsonl"}]


def test_not_found_message_names_the_folder_the_server_reads():
    session = Capture(_R({"detail": "로컬 청크 파일을 찾을 수 없습니다."}, status=404))
    with pytest.raises(m.IngestError) as info:
        m.save_vectors(session, "http://127.0.0.1:8001", "t", chunk_file="05_chunks.jsonl", dataset_id="ds", count=2,
                       data_type="guides", subdir="law")
    text = str(info.value)
    assert "LOCAL_CHUNK_ROOT/guides/law/" in text and "chunk_index_map.json" in text and "out\\<data_type>" in text
    session = Capture(_R({"detail": "x"}, status=404))
    with pytest.raises(m.IngestError, match="LOCAL_CHUNK_DIR"):
        m.save_vectors(session, "http://127.0.0.1:8001", "t", chunk_file="05_chunks.jsonl", dataset_id="ds", count=2)


# ── run() ─────────────────────────────────────────────────────────────────────

def test_dry_run_prints_where_the_server_reads(tmp_path):
    chunks = prepare(tmp_path)
    out: list[str] = []
    assert m.run(parse(chunks, "--dry-run"), log=out.append) == 0
    assert any("<LOCAL_CHUNK_ROOT>/guides/law/05_chunks.jsonl" in line and "data_type=guides" in line for line in out)
    out = []
    assert m.run(parse(chunks, "--dry-run", "--data-type", "none"), log=out.append) == 0
    assert any("LOCAL_CHUNK_DIR" in line and "옛 서버" in line for line in out)


class Server:
    """kb-ingest 가짜: 받은 요청을 기록하고 벡터 파일을 쓴다."""

    def __init__(self, chunks, vector_dir):
        self.chunks, self.vector_dir, self.requests = chunks, vector_dir, []

    def get(self, url, **kw):
        if url.endswith("/ready"):
            return _R({})
        if url.endswith("/openapi.json"):
            return _R({"paths": {m.SAVE_ROUTE: {}, m.LOAD_ROUTE: {}}})
        return _R({"result": {"config": {"params": {"vectors": {"dense": {"size": DIM}}, "sparse_vectors": {"sparse": {}}}}}})

    def post(self, url, **kw):
        self.requests.append(kw["json"])
        records = m.load_records(self.chunks)
        indexes = json.loads((self.chunks.parent / "chunk_index_map.json").read_text(encoding="utf-8"))["indexes"]
        pts = m.expected_points(records, indexes, LAW)
        lines = []
        for i, p in enumerate(pts):
            payload = {"source_type": "LAW", "site_id": 0, "flagged": False, "text": p.record["text"],
                       "title": p.record["title"], "locator": p.record["locator"],
                       "kb_source_id": 1, "kb_chunk_id": 100 + i}
            lines.append(json.dumps({"id": p.point_id, "payload": payload, "vector": {
                "dense": [0.1, 0.2, 0.3], "sparse": {"indices": [1], "values": [0.5]}}}))
        (self.vector_dir / "ds.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _R({"saved": len(pts), "dataset_id": "ds", "file": "ds.jsonl", "collection": "kb_guides", "model": "fake"})


def run_no_load(tmp_path, monkeypatch, *extra):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", "t")
    monkeypatch.setenv("QDRANT_API_KEY", "w")
    chunks = prepare(tmp_path)
    vdir = tmp_path / "vectors"
    vdir.mkdir()
    server = Server(chunks, vdir)
    code = m.run(parse(chunks, "--vector-dir", str(vdir), "--no-load", *extra), session=server, log=lambda s: None)
    return code, server


def test_run_sends_guides_law_by_default(tmp_path, monkeypatch):
    code, server = run_no_load(tmp_path, monkeypatch)
    assert code == 0
    assert server.requests == [{"dataset_id": "ds", "chunk_file": "05_chunks.jsonl", "data_type": "guides", "subdir": "law"}]


def test_run_custom_subdir_and_legacy_request(tmp_path, monkeypatch):
    _, server = run_no_load(tmp_path, monkeypatch, "--subdir", "law-2026")
    assert server.requests[0]["subdir"] == "law-2026" and server.requests[0]["data_type"] == "guides"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _, server = run_no_load(legacy, monkeypatch, "--data-type", "none")
    assert server.requests == [{"dataset_id": "ds", "chunk_file": "05_chunks.jsonl"}]
