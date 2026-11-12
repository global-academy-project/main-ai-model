"""ingest_law_vectors: chunk_index 대응표 배정(--update-index-map)과 장부 id 필수 검사.

서버·Qdrant 없이 가짜 세션과 임시 폴더로 돈다. 기존 tests/test_ingest_law_vectors.py 와 겹치지 않게 별도 파일로 둔다.
"""

from __future__ import annotations

import json

import pytest

from scripts import ingest_law_vectors as m

DIM = 3
LAW = "산업안전보건기준에관한규칙"


def record(n: int) -> dict:
    return {
        "natural_key": {"law_key": LAW, "article_key": f"제{n}조", "part": 0},
        "source_type": "LAW", "collection": "kb_guides", "site_id": 0, "flagged": False,
        "text": f"본문{n}", "embed_text": f"임베딩{n}", "locator": f"제{n}조", "title": "산업안전보건기준에 관한 규칙",
    }


def point(n: int, index: int | None = None) -> m.ExpectedPoint:
    return m.ExpectedPoint(record(n), m.point_id_for(LAW, n if index is None else index), f"제{n}조")


def write_vectors(path, points, ledger_for):
    lines = []
    for i, p in enumerate(points):
        payload = {"source_type": "LAW", "site_id": 0, "flagged": False, "text": p.record["text"],
                   "title": p.record["title"], "locator": p.record["locator"]}
        payload.update(ledger_for(i))
        lines.append(json.dumps({
            "id": p.point_id,
            "vector": {"dense": [0.1, 0.2, 0.3], "sparse": {"indices": [1], "values": [0.5]}},
            "payload": payload,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ids(source, chunk):
    return {"kb_source_id": source, "kb_chunk_id": chunk}


def check(tmp_path, ledgers, **kw):
    pts = [point(i) for i in range(len(ledgers))]
    path = tmp_path / "v.jsonl"
    write_vectors(path, pts, lambda i: ledgers[i])
    return m.check_vector_file(path, pts, DIM, **kw)


# ── 장부 id 검사 (check_vector_file) ───────────────────────────────────────────

def test_ids_present_pass(tmp_path):
    raw, problems = check(tmp_path, [ids(1, 100), ids(1, 101)], require_ledger_ids=True)
    assert len(raw) == 2 and problems == []


def test_null_ids_pass_when_not_required(tmp_path):
    assert check(tmp_path, [ids(None, None)])[1] == []


def test_null_ids_fail_when_required(tmp_path):
    _, problems = check(tmp_path, [ids(1, 100), ids(None, None)], require_ledger_ids=True)
    assert len(problems) == 1 and "--require-ledger-ids" in problems[0] and "제1조" in problems[0]


@pytest.mark.parametrize("bad", [0, -1, "1", 1.5, True])
def test_bad_values_reported_even_if_not_required(tmp_path, bad):
    _, problems = check(tmp_path, [ids(bad, 100)])
    assert len(problems) == 1 and "양의 정수나 null이 아님" in problems[0]


def test_only_one_side_and_duplicate_chunk_id(tmp_path):
    _, problems = check(tmp_path, [ids(1, None)])
    assert len(problems) == 1 and "하나만 있음" in problems[0]
    _, problems = check(tmp_path, [ids(1, 100), ids(1, 100)])
    assert len(problems) == 1 and "kb_chunk_id 100가 겹침" in problems[0]


def test_existing_text_check_still_works(tmp_path):
    pts = [point(0)]
    path = tmp_path / "v.jsonl"
    write_vectors(path, pts, lambda i: ids(1, 100))
    other = [m.ExpectedPoint({**record(0), "text": "다른 본문"}, pts[0].point_id, "제0조")]
    _, problems = m.check_vector_file(path, other, DIM, require_ledger_ids=True)
    assert len(problems) == 1 and "payload.text가 05와 다름" in problems[0]


# ── verify(): 장부 연결 부족 ────────────────────────────────────────────────────

class _R:
    def __init__(self, data, status=200):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


class _Q:
    """Qdrant 가짜: 점 payload 를 그대로 돌려준다."""

    def __init__(self, payloads):
        self.payloads = payloads            # {point_id: payload}

    def post(self, url, **kw):
        if url.endswith("/points/count"):
            return _R({"result": {"count": len(self.payloads)}})
        body = kw["json"]
        result = []
        for pid in body["ids"]:
            if pid in self.payloads:
                item = {"id": pid, "payload": self.payloads[pid]}
                if body.get("with_vector"):
                    item["vector"] = {"dense": [0.1] * DIM, "sparse": {"indices": [1], "values": [0.5]}}
                result.append(item)
        return _R({"result": result})


def full_payload(n, **ledger):
    r = record(n)
    return {"source_type": "LAW", "site_id": 0, "flagged": False, "text": r["text"], "title": r["title"],
            "locator": r["locator"], **ledger}


def test_verify_unlinked_is_note_by_default_and_problem_when_required():
    pts = [point(0), point(1)]
    payloads = {pts[0].point_id: full_payload(0, **ids(1, 100)), pts[1].point_id: full_payload(1)}
    problems, notes = m.verify(_Q(payloads), "http://q", "k", pts, embedding_dim=DIM)
    assert problems == [] and len(notes) == 1 and "1/2" in notes[0] and "doc_id" in notes[0]
    problems, notes = m.verify(_Q(payloads), "http://q", "k", pts, embedding_dim=DIM, require_ledger_ids=True)
    assert len(problems) == 1 and "1/2" in problems[0] and notes == []


def test_verify_all_linked_is_clean():
    pts = [point(0), point(1)]
    payloads = {p.point_id: full_payload(i, **ids(1, 100 + i)) for i, p in enumerate(pts)}
    assert m.verify(_Q(payloads), "http://q", "k", pts, embedding_dim=DIM, require_ledger_ids=True) == ([], [])


# ── 대응표: 배정·저장·불러오기 ───────────────────────────────────────────────────

def test_assign_indexes_only_adds_new_keys():
    recs = [record(1), record(2), record(3)]
    existing = {m.natural_key_str(recs[0]): 0, m.natural_key_str(recs[1]): 5}     # 6 이후는 비어 있다
    result = m.assign_indexes(recs, existing)
    assert result[m.natural_key_str(recs[0])] == 0 and result[m.natural_key_str(recs[1])] == 5   # 기존 번호 그대로
    assert result[m.natural_key_str(recs[2])] == 6                                              # 새 키는 최대+1
    assert existing == {m.natural_key_str(recs[0]): 0, m.natural_key_str(recs[1]): 5}           # 입력을 바꾸지 않는다


def test_assign_indexes_starts_at_zero_and_keeps_stale_keys():
    recs = [record(1), record(2)]
    assert sorted(m.assign_indexes(recs, {}).values()) == [0, 1]
    stale = {"옛법|제9조|0": 3}
    result = m.assign_indexes(recs, stale)
    assert result["옛법|제9조|0"] == 3 and sorted(result.values()) == [3, 4, 5]


def test_assign_indexes_rejects_duplicate_numbers():
    with pytest.raises(m.InputError):
        m.assign_indexes([record(1)], {"a|b|0": 1, "c|d|0": 1})


def test_save_and_load_roundtrip_with_backup(tmp_path):
    path = tmp_path / "chunk_index_map.json"
    m.save_index_map(path, LAW, {"k1": 0})
    assert not (tmp_path / "chunk_index_map.json.bak").exists()           # 처음엔 백업할 것이 없다
    m.save_index_map(path, LAW, {"k1": 0, "k2": 1})
    bak = json.loads((tmp_path / "chunk_index_map.json.bak").read_text(encoding="utf-8"))
    assert bak["indexes"] == {"k1": 0}                                     # 덮어쓰기 전 판이 남는다
    assert m.load_index_map(path, LAW) == {"k1": 0, "k2": 1}
    assert not list(tmp_path.glob("*.tmp"))                                # 임시 파일이 남지 않는다
    assert path.read_bytes().endswith(b"\n") and b"\r" not in path.read_bytes()


def test_load_index_map_missing_file(tmp_path):
    path = tmp_path / "chunk_index_map.json"
    with pytest.raises(m.InputError, match="--update-index-map"):
        m.load_index_map(path, LAW)
    assert m.load_index_map(path, LAW, allow_missing=True) == {}


def test_load_index_map_other_law_still_rejected(tmp_path):
    path = tmp_path / "chunk_index_map.json"
    m.save_index_map(path, "다른법", {"k": 0})
    with pytest.raises(m.InputError):
        m.load_index_map(path, LAW, allow_missing=True)


# ── 옵션 ─────────────────────────────────────────────────────────────────────

def test_options_defaults_and_conflicts():
    args = m.parse_args(["--dry-run"])
    assert args.require_ledger_ids is True and args.update_index_map is False
    assert m.parse_args(["--no-require-ledger-ids"]).require_ledger_ids is False
    assert m.parse_args(["--update-index-map"]).update_index_map is True
    with pytest.raises(SystemExit):
        m.parse_args(["--verify-only", "--update-index-map"])


# ── run() 전체 흐름 ───────────────────────────────────────────────────────────

class FakeServer:
    """kb-ingest + Qdrant 가짜. save 요청이 오면 vector_dir 에 벡터 파일을 쓴다(서버 동작 흉내)."""

    def __init__(self, tmp_path, chunks, *, ledger_for, vector_dir):
        self.chunks, self.ledger_for, self.vector_dir = chunks, ledger_for, vector_dir
        self.store: dict[str, dict] = {}
        self.calls: list[str] = []
        self.map_seen_by_server = None
        self.index_map_path = chunks.parent / "chunk_index_map.json"

    def get(self, url, **kw):
        self.calls.append("GET " + url.rsplit("/", 1)[-1])
        if url.endswith("/ready"):
            return _R({})
        if url.endswith("/openapi.json"):
            return _R({"paths": {m.SAVE_ROUTE: {}, m.LOAD_ROUTE: {}}})
        return _R({"result": {"config": {"params": {"vectors": {"dense": {"size": DIM}}, "sparse_vectors": {"sparse": {}}}}}})

    def post(self, url, **kw):
        if url.endswith(m.SAVE_ROUTE):
            self.calls.append("SAVE")
            self.map_seen_by_server = json.loads(self.index_map_path.read_text(encoding="utf-8"))["indexes"]
            records = m.load_records(self.chunks)
            pts = m.expected_points(records, self.map_seen_by_server, LAW)
            write_vectors(self.vector_dir / f"{kw['json']['dataset_id']}.jsonl", pts, self.ledger_for)
            return _R({"saved": len(pts), "dataset_id": kw["json"]["dataset_id"], "file": f"{kw['json']['dataset_id']}.jsonl",
                       "collection": "kb_guides", "model": "fake"})
        if url.endswith("/points/count"):
            return _R({"result": {"count": len(self.store)}})
        body = kw["json"]
        result = []
        for pid in body["ids"]:
            if pid in self.store:
                item = {"id": pid, "payload": self.store[pid]["payload"]}
                if body.get("with_vector"):
                    item["vector"] = self.store[pid]["vector"]
                result.append(item)
        return _R({"result": result})

    def put(self, url, **kw):
        self.calls.append("PUT")
        for p in kw["json"]["points"]:
            self.store[p["id"]] = p
        return _R({"status": "ok"})


def setup_run(tmp_path, monkeypatch, *, n=3, ledger_for=lambda i: ids(1, 100 + i), map_keys=None):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", "t")
    monkeypatch.setenv("QDRANT_API_KEY", "w")
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "r")
    chunks = tmp_path / "05_chunks.jsonl"
    chunks.write_text("".join(json.dumps(record(i)) + "\n" for i in range(n)), encoding="utf-8")
    (tmp_path / "06_validation.json").write_text(json.dumps({"ok": True, "summary": {"chunks": n}}), encoding="utf-8")
    if map_keys is not None:
        m.save_index_map(tmp_path / "chunk_index_map.json", LAW, map_keys)
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    return chunks, vector_dir, FakeServer(tmp_path, chunks, ledger_for=ledger_for, vector_dir=vector_dir)


def args_for(chunks, vector_dir, *extra):
    return m.parse_args(["--chunks", str(chunks), "--vector-dir", str(vector_dir), "--dataset-id", "ds",
                         "--embedding-dim", str(DIM), *extra])


def keys(n):
    return {m.natural_key_str(record(i)): i for i in range(n)}


def test_run_dry_run_reports_new_numbers_without_writing(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch)                       # 대응표 없음
    out: list[str] = []
    code = m.run(args_for(chunks, vdir, "--update-index-map", "--dry-run"), session=server, log=out.append)
    assert code == 0 and any("새 번호 3개" in line for line in out)
    assert not (tmp_path / "chunk_index_map.json").exists() and server.calls == []


def test_run_without_flag_still_requires_complete_map(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys={m.natural_key_str(record(0)): 0})
    with pytest.raises(m.InputError, match="--update-index-map"):
        m.run(args_for(chunks, vdir), session=server, log=lambda s: None)
    assert server.calls == []


def test_run_update_map_is_written_before_server_is_called(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys={m.natural_key_str(record(0)): 0})
    out: list[str] = []
    code = m.run(args_for(chunks, vdir, "--update-index-map"), session=server, log=out.append)
    assert code == 0
    assert server.map_seen_by_server == keys(3)                       # 서버가 읽을 때 이미 새 번호가 들어 있었다
    assert (tmp_path / "chunk_index_map.json.bak").exists()
    assert len(server.store) == 3 and any("검증 통과" in line for line in out)


def test_run_full_flow_with_ids_uploads(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys=keys(3))
    assert m.run(args_for(chunks, vdir), session=server, log=lambda s: None) == 0
    assert server.calls.count("PUT") == 1 and len(server.store) == 3


def test_run_refuses_to_upload_points_without_ledger_ids(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys=keys(3), ledger_for=lambda i: ids(None, None))
    with pytest.raises(m.IngestError, match="장부 id"):
        m.run(args_for(chunks, vdir), session=server, log=lambda s: None)
    assert "PUT" not in server.calls and server.store == {}           # Qdrant에는 아무것도 올라가지 않았다


def test_run_can_opt_out_of_ledger_requirement(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys=keys(3), ledger_for=lambda i: ids(None, None))
    out: list[str] = []
    assert m.run(args_for(chunks, vdir, "--no-require-ledger-ids"), session=server, log=out.append) == 0
    assert len(server.store) == 3 and any("[참고] 장부 연결" in line for line in out)


def test_run_no_load_skips_ledger_requirement_and_upload(tmp_path, monkeypatch):
    chunks, vdir, server = setup_run(tmp_path, monkeypatch, map_keys=keys(3), ledger_for=lambda i: ids(None, None))
    assert m.run(args_for(chunks, vdir, "--no-load"), session=server, log=lambda s: None) == 0
    assert "PUT" not in server.calls and (vdir / "ds.jsonl").is_file()
