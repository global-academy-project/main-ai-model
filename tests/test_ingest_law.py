"""scripts/ingest_law.py 시험. DB·네트워크 없이 돈다 (가짜 세션·가짜 연결)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts import ingest_law as il

UUID5 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")  # V1 CHECK와 같은 식
LAW = "산업안전보건기준에관한규칙"


def record(article: str, part: int = 0, **over: Any) -> dict[str, Any]:
    base = {
        "natural_key": {"law_key": LAW, "article_key": article, "part": part},
        "article_label": f"제{article}조",
        "text": f"제{article}조(제목) 본문",
        "embed_text": f"산업안전보건기준에 관한 규칙 > 제1편\n제{article}조(제목) 본문",
        "title": "산업안전보건기준에 관한 규칙",
        "locator": f"제{article}조",
        "source_type": "LAW",
        "collection": "kb_guides",
        "site_id": 0,
        "flagged": False,
        "year": 2025,
        "hazard": [],
        "work_type": [],
        "cas": [],
        "url": None,
        "file_path": "raw/law.hwpx",
        "law_meta": {"promulgation_date": "2025-09-01", "source_sha256": "a" * 64},
        "kb_source_id": None,
        "kb_chunk_id": None,
        "point_id": None,
    }
    base.update(over)
    return base


def records(n: int = 5) -> list[dict[str, Any]]:
    return [record(str(i + 1)) for i in range(n)]


def write_validation(path: Path, *, ok: bool = True, chunks: int | None = 5) -> Path:
    summary = {"chunks": chunks} if chunks is not None else {}
    path.write_text(json.dumps({"ok": ok, "summary": summary}), encoding="utf-8")
    return path


# ---------------------------------------------------------------- id 규칙


def test_point_id_is_uuid5_of_documented_formula():
    import uuid

    pid = il.point_id_for(LAW, 7)
    assert pid == str(uuid.uuid5(uuid.NAMESPACE_URL, f"LAW:{LAW}:7"))
    assert UUID5.match(pid)
    assert il.point_id_for(LAW, 7) == pid
    assert il.point_id_for(LAW, 8) != pid


def test_point_id_matches_init_qdrant_formula():
    pytest.importorskip("qdrant_client")
    from scripts.init_qdrant import point_id

    assert il.point_id_for(LAW, 12) == point_id("LAW", LAW, 12)


# ---------------------------------------------------------------- 대응표


def test_assign_indexes_numbers_in_document_order():
    recs = records(4)
    assert il.assign_indexes(recs, {}) == {il.natural_key_str(r): i for i, r in enumerate(recs)}


def test_assign_indexes_keeps_existing_and_appends_new_after_max():
    recs = records(3)
    existing = {il.natural_key_str(recs[0]): 0, il.natural_key_str(recs[2]): 9}
    result = il.assign_indexes(recs, existing)
    assert result[il.natural_key_str(recs[0])] == 0
    assert result[il.natural_key_str(recs[2])] == 9
    assert result[il.natural_key_str(recs[1])] == 10  # 번호가 밀리지도, 재사용되지도 않는다


def test_assign_indexes_never_reuses_numbers_of_removed_keys():
    recs = records(3)
    first = il.assign_indexes(recs, {})
    after = il.assign_indexes([recs[0], recs[2], record("99")], first)  # 제2조가 사라짐
    assert after[il.natural_key_str(record("99"))] == 3
    assert after[il.natural_key_str(recs[1])] == 1  # 남겨 둔다


def test_assign_indexes_rejects_duplicate_numbers():
    with pytest.raises(il.InputError):
        il.assign_indexes(records(2), {"a": 1, "b": 1})


def test_index_map_roundtrip_and_law_mismatch(tmp_path):
    path = tmp_path / "map.json"
    il.save_index_map(path, LAW, {"k|1|0": 0, "k|2|0": 1})
    assert il.load_index_map(path, LAW) == {"k|1|0": 0, "k|2|0": 1}
    assert il.load_index_map(tmp_path / "none.json", LAW) == {}
    with pytest.raises(il.InputError):
        il.load_index_map(path, "다른법령")
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------- 청크 파일 검사


def test_check_records_accepts_good_input(tmp_path):
    assert il.check_records(records(), write_validation(tmp_path / "v.json")) == LAW


@pytest.mark.parametrize(
    "mutate",
    [
        lambda recs: recs[1].update(natural_key=dict(recs[0]["natural_key"])),                   # natural_key 중복
        lambda recs: recs[1].update(source_type="KOSHA_GUIDE"),
        lambda recs: recs[1].update(collection="kb_accidents"),
        lambda recs: recs[1].update(site_id=3),
        lambda recs: recs[1].update(flagged=True),
        lambda recs: recs[1].update(text="  "),
        lambda recs: recs[1].update(embed_text=""),
        lambda recs: recs[1].update(locator="x" * 101),
        lambda recs: recs[1]["natural_key"].update(law_key="다른법령"),                            # 법령이 둘
        lambda recs: recs[1]["law_meta"].update(source_sha256="b" * 64),                         # 원본이 둘
    ],
)
def test_check_records_rejects_bad_records(tmp_path, mutate):
    recs = records()
    mutate(recs)
    with pytest.raises(il.InputError):
        il.check_records(recs, write_validation(tmp_path / "v.json"))


def test_check_records_requires_passing_validation_and_matching_count(tmp_path):
    with pytest.raises(il.InputError):
        il.check_records(records(), write_validation(tmp_path / "v.json", ok=False))
    with pytest.raises(il.InputError):
        il.check_records(records(), write_validation(tmp_path / "v.json", chunks=6))
    with pytest.raises(il.InputError):
        il.check_records(records(), tmp_path / "missing.json")
    with pytest.raises(il.InputError):
        il.check_records([], write_validation(tmp_path / "v.json"))


def test_load_records_reports_bad_json(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"a": 1}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(il.InputError, match="3번째"):
        il.load_records(path)


# ---------------------------------------------------------------- 요청 만들기


def test_build_items_uses_ledger_ids_and_matches_ingest_contract():
    recs = records(40)
    indexes = il.assign_indexes(recs, {})
    chunk_ids = {i: 100 + i for i in range(40)}
    items = il.build_items(recs, indexes, LAW, 7, chunk_ids)
    assert items[3]["point_id"] == il.point_id_for(LAW, 3)
    assert items[3]["payload"]["kb_source_id"] == 7
    assert items[3]["payload"]["kb_chunk_id"] == 103
    assert items[3]["text"] == recs[3]["embed_text"]      # 임베딩 입력
    assert items[3]["payload"]["text"] == recs[3]["text"]  # 인용 원문
    assert len({item["point_id"] for item in items}) == 40

    pytest.importorskip("fastapi")
    from app.ingest_main import IngestRequest

    for batch in il.batched(items):
        IngestRequest(collection="kb_guides", chunks=list(batch))  # 서버 검증을 통과한다


def test_batched_default_matches_embedding_server_limit():
    # 임베딩 서버 EMBEDDING_MAX_BATCH_SIZE 기본 8 -- 넘으면 HTTP 422 -> kb-ingest 502
    assert il.BATCH_SIZE == 8
    sizes = [len(b) for b in il.batched(list(range(695)))]
    assert len(sizes) == 87 and sizes[-1] == 7 and max(sizes) == 8
    assert sum(sizes) == 695


def test_batched_never_exceeds_ingest_limit():
    assert il.MAX_BATCH_SIZE == 32
    sizes = [len(b) for b in il.batched(list(range(695)), il.MAX_BATCH_SIZE)]
    assert len(sizes) == 22 and sizes[-1] == 23 and max(sizes) == 32
    assert sum(sizes) == 695


def test_batch_size_option(tmp_path):
    chunks = str(tmp_path / "05_chunks.jsonl")
    assert il.parse_args(["--chunks", chunks]).batch_size == 8
    assert il.parse_args(["--chunks", chunks, "--batch-size", "32"]).batch_size == 32
    assert il.parse_args(["--chunks", chunks, "--batch-size", "1"]).batch_size == 1
    for bad in ("0", "33", "-1"):
        with pytest.raises(SystemExit):
            il.parse_args(["--chunks", chunks, "--batch-size", bad])


class RunLedger:
    """run()이 쓰는 장부 메서드만 흉내 낸다 (DB 없음)."""

    def __init__(self) -> None:
        self.statuses: list[str] = []

    def find_source(self, external_id):
        return None

    def upsert_source(self, meta):
        return 1

    def upsert_chunks(self, source_id, rows):
        return {row["chunk_index"]: 1000 + row["chunk_index"] for row in rows}

    def set_status(self, source_id, status, **kwargs):
        self.statuses.append(status)


def run_with_recorded_batches(tmp_path, monkeypatch, extra_args, *, n=20):
    pytest.importorskip("requests")
    chunks = tmp_path / "05_chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records(n)), encoding="utf-8")
    write_validation(tmp_path / "06_validation.json", chunks=n)
    sizes: list[int] = []

    def fake_post(session, url, token, batch):
        sizes.append(len(batch))
        raise il.IngestError("stop")  # 첫 요청 크기만 보고 멈춘다 (검증 단계로 가지 않음)

    monkeypatch.setenv("KB_INGEST_API_TOKEN", "t")
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "k")
    monkeypatch.setattr(il, "connect_ledger", lambda: RunLedger())
    monkeypatch.setattr(il, "post_batch", fake_post)
    code = il.run(il.parse_args(["--chunks", str(chunks), *extra_args]))
    return code, sizes


def test_run_sends_default_batches_of_eight(tmp_path, monkeypatch):
    code, sizes = run_with_recorded_batches(tmp_path, monkeypatch, [])
    assert code == 1 and sizes == [8]


def test_run_honors_batch_size_option(tmp_path, monkeypatch):
    code, sizes = run_with_recorded_batches(tmp_path, monkeypatch, ["--batch-size", "5"])
    assert code == 1 and sizes == [5]


def test_dry_run_reports_request_count_with_batch_size(tmp_path, monkeypatch, capsys):
    chunks = tmp_path / "05_chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records(20)), encoding="utf-8")
    write_validation(tmp_path / "06_validation.json", chunks=20)
    assert il.run(il.parse_args(["--chunks", str(chunks), "--dry-run"])) == 0
    assert "요청 3번 (8개씩)" in capsys.readouterr().out
    assert il.run(il.parse_args(["--chunks", str(chunks), "--dry-run", "--batch-size", "32"])) == 0
    assert "요청 1번 (32개씩)" in capsys.readouterr().out


# ---------------------------------------------------------------- post_batch


class Resp:
    def __init__(self, code: int, body: dict[str, Any] | None = None, text: str = "") -> None:
        self.status_code = code
        self._body = body or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._body


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Resp:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


ITEMS = [{"point_id": "x", "text": "t", "payload": {}}] * 3


def test_post_batch_success_sends_collection_and_bearer():
    session = FakeSession([Resp(200, {"indexed": 3})])
    assert il.post_batch(session, "http://h:8001/", "tok", ITEMS, sleep=lambda s: None) == 3
    call = session.calls[0]
    assert call["url"] == "http://h:8001/ingest/chunks"
    assert call["headers"] == {"Authorization": "Bearer tok"}
    assert call["json"]["collection"] == "kb_guides" and len(call["json"]["chunks"]) == 3


def test_post_batch_retries_server_errors_and_connection_errors():
    sleeps: list[float] = []
    session = FakeSession([Resp(502, text="x"), ConnectionError("down"), Resp(200, {"indexed": 3})])
    assert il.post_batch(session, "http://h", "t", ITEMS, sleep=sleeps.append) == 3
    assert len(session.calls) == 3 and sleeps == [5.0, 10.0]


def test_post_batch_gives_up_after_retries():
    session = FakeSession([Resp(503, text="no")] * 3)
    with pytest.raises(il.IngestError, match="계속 실패"):
        il.post_batch(session, "http://h", "t", ITEMS, sleep=lambda s: None)
    assert len(session.calls) == 3


def test_post_batch_does_not_retry_client_errors():
    session = FakeSession([Resp(422, text="bad payload")])
    with pytest.raises(il.IngestError, match="거부"):
        il.post_batch(session, "http://h", "t", ITEMS, sleep=lambda s: None)
    assert len(session.calls) == 1


def test_post_batch_rejects_count_mismatch():
    with pytest.raises(il.IngestError, match="적재 수"):
        il.post_batch(FakeSession([Resp(200, {"indexed": 2})]), "http://h", "t", ITEMS, sleep=lambda s: None)


# ---------------------------------------------------------------- 검증


class FakeLedger:
    def __init__(self, count: int) -> None:
        self.count = count

    def count_chunks(self, source_id: int) -> int:
        return self.count


class QdrantSession:
    def __init__(self, count: int, points: list[dict[str, Any]]) -> None:
        self.count, self.points = count, points

    def post(self, url: str, **kwargs: Any) -> Resp:
        if url.endswith("/points/count"):
            return Resp(200, {"result": {"count": self.count}})
        return Resp(200, {"result": self.points})


def good_point(pid: str, chunk_id: int, source_id: int = 1, dim: int = 1024) -> dict[str, Any]:
    return {
        "id": pid,
        "payload": {"kb_source_id": source_id, "kb_chunk_id": chunk_id, "source_type": "LAW",
                    "flagged": False, "text": "본문"},
        "vector": {"dense": [0.1] * dim},
    }


def run_verify(ledger_count=5, qdrant_count=5, points=None, full=True, expected=5):
    samples = [{"point_id": "p1", "chunk_id": 11, "locator": "제1조"}]
    points = [good_point("p1", 11)] if points is None else points
    return il.verify(FakeLedger(ledger_count), QdrantSession(qdrant_count, points), "http://q", "key",
                     source_id=1, expected=expected, full=full, samples=samples, embedding_dim=1024)


def test_verify_passes_on_consistent_state():
    assert run_verify() == []


def test_verify_detects_each_kind_of_drift():
    assert any("장부 청크 수" in p for p in run_verify(ledger_count=4))
    assert any("Qdrant 점 수" in p for p in run_verify(qdrant_count=4))
    assert any("Qdrant 점 수" in p for p in run_verify(qdrant_count=6))          # 전체 적재에서는 남는 점도 문제
    assert any("없습니다" in p for p in run_verify(points=[]))
    assert any("장부와 다릅니다" in p for p in run_verify(points=[good_point("p1", 99)]))
    assert any("장부와 다릅니다" in p for p in run_verify(points=[good_point("p1", 11, source_id=2)]))
    assert any("차원" in p for p in run_verify(points=[good_point("p1", 11, dim=3)]))
    bad = good_point("p1", 11)
    bad["payload"]["text"] = " "
    assert any("text" in p for p in run_verify(points=[bad]))


def test_verify_partial_run_allows_more_than_expected():
    assert run_verify(ledger_count=9, qdrant_count=9, full=False) == []
    assert run_verify(ledger_count=2, qdrant_count=9, full=False, expected=5) != []


def test_sample_positions():
    assert il.sample_positions(2) == [0, 1]
    assert il.sample_positions(695) == [0, 347, 694]


# ---------------------------------------------------------------- SQL 자리표시자


class RecordingCursor:
    def __init__(self, log: list[tuple[str, tuple]]) -> None:
        self.log = log
        self.lastrowid = 1
        self.last_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.last_sql = sql
        self.log.append((sql, params))

    def fetchone(self):
        return (0,) if "COUNT(*)" in self.last_sql else None

    def fetchall(self):
        return []


class RecordingConn:
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []

    def cursor(self):
        return RecordingCursor(self.log)

    def commit(self):
        pass


def test_every_sql_statement_has_matching_placeholder_count():
    conn = RecordingConn()
    ledger = il.Ledger(conn)
    meta = il.SourceMeta(LAW, "제목", "raw/a.hwpx", "a" * 64, None, None, None)
    ledger.find_source(LAW)
    ledger.upsert_source(meta)
    ledger.upsert_chunks(1, [{"chunk_index": 0, "point_id": "p", "locator": "제1조", "text": "t"}])
    ledger.set_status(1, "FAILED", error="e")
    ledger.count_chunks(1)
    ledger.chunk_id_map(1)
    assert len(conn.log) >= 7
    for sql, params in conn.log:
        assert sql.count("%s") == len(params), sql


def test_parse_args_defaults_are_next_to_chunks_file(tmp_path):
    args = il.parse_args(["--chunks", str(tmp_path / "05_chunks.jsonl")])
    assert args.validation == tmp_path / "06_validation.json"
    assert args.index_map == tmp_path / "chunk_index_map.json"
    assert args.limit == 0 and not args.dry_run
