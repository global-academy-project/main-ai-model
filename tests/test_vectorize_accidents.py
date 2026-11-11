"""scripts/vectorize_accidents.py 단위 시험. kb-ingest는 가짜 세션으로 흉내 낸다.

가짜 서버는 ingest_main.py(2026-11-10 16:14판)의 /ingest/vectors/local/file (data_type=accidents) 동작을 따른다:
<root>/accidents/<subdir>/<chunk_file>을 읽고, validation_file이 오면 ok·dataset=sif·청크 수를 확인하고,
embed_text를 임베딩해 <vector_dir>/<dataset_id>.jsonl 에 {id, vector, payload}를 쓴다.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts import vectorize_accidents as va
from scripts.ingest_law_vectors import IngestError, InputError

DIM = 8
TOKEN = "t-ingest"
PAYLOAD_FIELDS = (
    "source_type", "site_id", "flagged", "title", "locator", "url", "text",
    "severity", "accident_type", "hazard", "work_type", "cas", "ghs", "year",
    "external_id", "chunk_index", "natural_key", "license", "publisher",
    "content_sha256", "tier", "cate", "disaster_type", "situation",
)


def make_record(i: int, **over):
    ext = f"SIF-{i:016x}"
    record = {
        "natural_key": {"dataset": "sif", "record": ext, "section": "case", "part": 0},
        "external_id": ext,
        "chunk_index": 0,
        "text": f"[재해개요]\n사례 {i} 개요\n\n[재해유발요인]\n요인 {i}",
        "embed_text": f"SIF > 공사 > 작업 > 추락\n[재해개요]\n사례 {i} 개요",
        "title": f"작업 {i} – 추락 (2019)",
        "locator": "재해개요·유발요인·감소대책(예시)",
        "source_type": "SIF",
        "collection": "kb_accidents",
        "site_id": 0,
        "flagged": False,
        "year": 2019,
        "severity": "FATAL",
        "accident_type": ["FALL"],
        "hazard": [],
        "work_type": ["HEIGHT"],
        "cas": [],
        "ghs": [],
        "url": "https://www.data.go.kr/data/15161362/openapi.do",
        "license": "이용허락범위 제한 없음",
        "publisher": "한국산업안전보건공단",
        "content_sha256": "a" * 64,
        "tier": "A",
        "cate": "전업종",
        "disaster_type": "추락",
        "situation": None,
        "kb_source_id": None,
        "kb_chunk_id": None,
        "point_id": None,
    }
    record.update(over)
    return record


def make_records(n: int = 6):
    return [make_record(i) for i in range(1, n + 1)]


def write_chunks(folder: Path, records, *, validation="auto", name="05_chunks.jsonl"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    if validation == "auto":
        validation = {"ok": True, "dataset": "sif", "summary": {"chunks": len(records)}}
    if validation is not None:
        (folder / "06_validation.json").write_text(json.dumps(validation), encoding="utf-8")
    return path


class Resp:
    def __init__(self, status: int = 200, data=None, text: str = ""):
        self.status_code = status
        self._data = data
        self.text = text or (json.dumps(data, ensure_ascii=False) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("JSON 아님")
        return self._data


class FakeSession:
    def __init__(self, root: Path, vector_dir: Path, subdir: str = "law"):
        self.root = root
        self.vector_dir = vector_dir
        self.subdir = subdir
        self.script = None
        self.attempts: defaultdict[str, int] = defaultdict(int)
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[dict] = []
        self.timeouts: list[float] = []
        self.saved_delta = 0
        self.omit_sparse = False
        self.empty_sparse = False
        self.wrong_file = False
        self.wrong_collection = False
        self.wrong_validation = False
        self.tamper = None  # fn(payload) -> None
        self.no_write = False
        self.ready_status = 200
        self.paths = {va.SAVE_ROUTE: {}}
        self.schema_has_data_type = True
        self.openapi_status = 200
        self.connect_error = False

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        if self.connect_error:
            raise ConnectionError("down")
        if url.endswith("/ready"):
            return Resp(self.ready_status, {"status": "ok"})
        if url.endswith("/openapi.json"):
            if self.openapi_status != 200:
                return Resp(self.openapi_status)
            props = {"dataset_id": {}, "chunk_file": {}, "subdir": {}}
            if self.schema_has_data_type:
                props["data_type"] = {}
            return Resp(200, {"paths": self.paths, "components": {"schemas": {"LocalChunkFileRequest": {"properties": props}}}})
        return Resp(404)

    def post(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append(("POST", url))
        self.timeouts.append(timeout)
        self.bodies.append(json)
        self.attempts["save"] += 1
        if (headers or {}).get("Authorization") != f"Bearer {TOKEN}":
            return Resp(401, {"detail": "unauthorized"})
        if self.script:
            action = self.script("save", self.attempts["save"])
            if isinstance(action, Exception):
                raise action
            if isinstance(action, int):
                return Resp(action, None, text="서버 메시지")
        body = json
        folder = self.root / "accidents" / body.get("subdir", "law")
        chunk_path = folder / body["chunk_file"]
        if body.get("data_type") != "accidents" or not chunk_path.is_file():
            return Resp(404, {"detail": "로컬 청크 파일을 찾을 수 없습니다."})
        validation = None
        if body.get("validation_file"):
            vpath = folder / body["validation_file"]
            if not vpath.is_file():
                return Resp(422, {"detail": "검증 파일을 찾을 수 없습니다"})
            validation = __import__("json").loads(vpath.read_text(encoding="utf-8"))
            if validation.get("ok") is not True or validation.get("dataset") != "sif":
                return Resp(422, {"detail": "검증 실패"})
        records = [__import__("json").loads(x) for x in chunk_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if validation is not None and validation["summary"]["chunks"] != len(records):
            return Resp(422, {"detail": "청크 수가 검증 결과와 일치하지 않습니다"})
        lines = []
        for record in records:
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"SIF:{record['external_id']}:{record['chunk_index']}"))
            payload = {f: record[f] for f in PAYLOAD_FIELDS if f in record}
            payload["kb_source_id"] = record.get("kb_source_id")
            payload["kb_chunk_id"] = record.get("kb_chunk_id")
            if self.tamper:
                self.tamper(payload)
            vector = {"dense": [0.1] * DIM}
            if not self.omit_sparse:
                vector["sparse"] = {"indices": [], "values": []} if self.empty_sparse else {"indices": [1, 2], "values": [1.0, 1.0]}
            lines.append({"id": pid, "vector": vector, "payload": payload})
        file_name = f"{body['dataset_id']}.jsonl"
        if not self.no_write:
            self.vector_dir.mkdir(parents=True, exist_ok=True)
            (self.vector_dir / file_name).write_text(
                "".join(__import__("json").dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8", newline="\n"
            )
        return Resp(200, {
            "saved": len(lines) + self.saved_delta,
            "dataset_id": body["dataset_id"],
            "file": "other.jsonl" if self.wrong_file else file_name,
            "collection": "kb_guides" if self.wrong_collection else "kb_accidents",
            "model": "BAAI/bge-m3",
            "validation": ("structural" if self.wrong_validation else "pipeline") if validation is not None else (
                "pipeline" if self.wrong_validation else "structural"),
        })


class Boom:
    def __getattr__(self, name):
        raise AssertionError(f"접속하면 안 됩니다: {name}")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    for name in ("KB_INGEST_URL", "EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)


NOW = datetime(2026, 11, 10, 16, 0, 0)


def ticking():
    state = {"t": NOW}

    def clock():
        state["t"] += timedelta(seconds=1)
        return state["t"]

    return clock


def setup(tmp_path: Path, records=None, **kw):
    records = records if records is not None else make_records()
    folder = tmp_path / "out" / "accidents" / "law"
    chunks = write_chunks(folder, records, **kw)
    vec = tmp_path / "vectors"
    return chunks, vec, FakeSession(tmp_path / "out", vec), records


def args_for(chunks: Path, vec: Path, *extra: str):
    return va.parse_args(["--chunks", str(chunks), "--vector-dir", str(vec), "--embedding-dim", str(DIM), *extra])


def go(args, session, logs=None, **kw):
    out = logs if logs is not None else []
    return va.run(args, session=session, sleep=lambda s: None, clock=ticking(), log=out.append, **kw)


# ---------------------------------------------------------------- id 규칙


def test_point_id_formula():
    assert va.point_id_for("SIF-4080de0545ed5df6", 0) == "bd15f973-7139-5a22-b1c7-22d95c225b3d"  # 실제 첫 사례(11/10 파일)
    assert va.point_id_for("a", 1) == str(uuid.uuid5(uuid.NAMESPACE_URL, "SIF:a:1"))
    assert va.point_id_for("a", 1) != va.point_id_for("a", 2)
    assert va.point_id_for("a", 0) != str(uuid.uuid5(uuid.NAMESPACE_URL, "LAW:a:0"))


# ---------------------------------------------------------------- 청크 검사


def test_check_records_ok():
    points = va.check_records(make_records(3))
    assert len(points) == 3 and points[0].label == "SIF-0000000000000001/0"


def test_check_records_accepts_missing_optional_fields():
    record = make_record(1)
    for field in ("year", "severity", "url", "kb_source_id", "kb_chunk_id", "situation"):
        record.pop(field, None)
    record["year"] = None
    assert len(va.check_records([record])) == 1


def test_check_records_empty_and_too_many():
    with pytest.raises(InputError, match="청크가 없습니다"):
        va.check_records([])
    with pytest.raises(InputError, match="최대 5000건"):
        va.check_records([make_record(1)] * 5001)


@pytest.mark.parametrize("name,mutate,match", [
    ("natural_key 없음", lambda r: r.pop("natural_key"), "natural_key가 없습니다"),
    ("dataset", lambda r: r["natural_key"].update(dataset="law"), "SIF 청크의 natural_key"),
    ("record 불일치", lambda r: r["natural_key"].update(record="SIF-x"), "SIF 청크의 natural_key"),
    ("section 빈 값", lambda r: r["natural_key"].update(section=" "), "SIF 청크의 natural_key"),
    ("part bool", lambda r: r["natural_key"].update(part=True), "SIF 청크의 natural_key"),
    ("part 음수", lambda r: r["natural_key"].update(part=-1), "SIF 청크의 natural_key"),
    ("external_id 없음", lambda r: r.pop("external_id"), "SIF 청크의 natural_key"),
    ("chunk_index bool", lambda r: r.update(chunk_index=False), "SIF 청크의 natural_key"),
    ("chunk_index 음수", lambda r: r.update(chunk_index=-1), "SIF 청크의 natural_key"),
    ("chunk_index 문자", lambda r: r.update(chunk_index="0"), "SIF 청크의 natural_key"),
    ("source_type", lambda r: r.update(source_type="LAW"), "SIF 청크의 natural_key"),
    ("collection", lambda r: r.update(collection="kb_guides"), "kb_accidents이 아닙니다"),
    ("external_id 김", lambda r: (r.update(external_id="가" * 101), r["natural_key"].update(record="가" * 101)), "100자"),
    ("text 빈 값", lambda r: r.update(text="  "), "text가 비어"),
    ("embed_text 없음", lambda r: r.pop("embed_text"), "embed_text가 비어"),
    ("site_id 음수", lambda r: r.update(site_id=-1), "site_id"),
    ("site_id bool", lambda r: r.update(site_id=True), "site_id"),
    ("flagged 문자", lambda r: r.update(flagged="false"), "flagged"),
    ("title 빈 값", lambda r: r.update(title=""), "title 값이 비어"),
    ("locator 없음", lambda r: r.pop("locator"), "locator 값이 비어"),
    ("year 문자", lambda r: r.update(year="2019"), "year"),
    ("year bool", lambda r: r.update(year=True), "year"),
    ("severity 숫자", lambda r: r.update(severity=1), "severity"),
    ("accident_type None", lambda r: r.update(accident_type=None), "accident_type는 문자열 배열"),
    ("hazard 숫자 원소", lambda r: r.update(hazard=[1]), "hazard는 문자열 배열"),
    ("ghs 없음", lambda r: r.pop("ghs"), "ghs는 문자열 배열"),
])
def test_check_records_rejects(name, mutate, match):
    records = make_records(3)
    mutate(records[1])
    with pytest.raises(InputError, match=match):
        va.check_records(records)


def test_check_records_duplicates():
    a, b = make_record(1), make_record(2)
    dup_natural = dict(a, external_id=a["external_id"])
    with pytest.raises(InputError, match="natural_key가 중복"):
        va.check_records([a, dup_natural])
    same_index = make_record(2, chunk_index=0)
    same_index["natural_key"] = dict(b["natural_key"], section="other")
    with pytest.raises(InputError, match="external_id/chunk_index가 중복"):
        va.check_records([b, same_index])


def test_check_records_multi_chunk_per_source_ok():
    a = make_record(1)
    b = make_record(1, chunk_index=1)
    b["natural_key"] = dict(a["natural_key"], part=1)
    points = va.check_records([a, b])
    assert points[0].point_id != points[1].point_id


# ---------------------------------------------------------------- 06 검사


def write_validation(tmp_path, data):
    p = tmp_path / "06_validation.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_check_validation_ok(tmp_path):
    va.check_validation(write_validation(tmp_path, {"ok": True, "dataset": "sif", "summary": {"chunks": 5}}), 5)


@pytest.mark.parametrize("data,match", [
    ({"ok": False, "dataset": "sif", "summary": {"chunks": 5}}, "ok가 true가 아닙니다"),
    ({"ok": "true", "dataset": "sif", "summary": {"chunks": 5}}, "ok가 true가 아닙니다"),
    ({"ok": True, "dataset": "law", "summary": {"chunks": 5}}, "dataset이 sif가 아닙니다"),
    ({"ok": True, "summary": {"chunks": 5}}, "dataset이 sif가 아닙니다"),
    ({"ok": True, "dataset": "sif", "summary": {"chunks": 6}}, "청크 수가 검증 결과와 다릅니다"),
    ({"ok": True, "dataset": "sif", "summary": {"chunks": True}}, "청크 수가 검증 결과와 다릅니다"),
    ({"ok": True, "dataset": "sif", "summary": {}}, "청크 수가 검증 결과와 다릅니다"),
    ([], "ok가 true가 아닙니다"),
])
def test_check_validation_rejects(tmp_path, data, match):
    with pytest.raises(InputError, match=match):
        va.check_validation(write_validation(tmp_path, data), 5)


def test_check_validation_unreadable(tmp_path):
    p = tmp_path / "06_validation.json"
    p.write_text("{", encoding="utf-8")
    with pytest.raises(InputError, match="JSON이 아닙니다"):
        va.check_validation(p, 5)
    with pytest.raises(InputError, match="읽을 수 없거나"):
        va.check_validation(tmp_path / "none.json", 5)


def test_default_dataset_id():
    assert va.default_dataset_id(lambda: datetime(2026, 11, 10)) == "sif-2026-11"
    assert va.default_dataset_id(lambda: datetime(2027, 1, 2)) == "sif-2027-01"


# ---------------------------------------------------------------- 사전 점검


def pre(s, log=lambda *_: None):
    va.preflight(s, "http://127.0.0.1:8001", log=log)


def test_preflight_ok(tmp_path):
    _, _, s, _ = setup(tmp_path)
    pre(s)


def test_preflight_missing_route(tmp_path):
    _, _, s, _ = setup(tmp_path)
    s.paths = {}
    with pytest.raises(InputError, match="다시 빌드"):
        pre(s)


def test_preflight_old_schema_without_data_type(tmp_path):
    _, _, s, _ = setup(tmp_path)
    s.schema_has_data_type = False
    with pytest.raises(InputError, match="data_type이 없는 옛 판"):
        pre(s)


def test_preflight_unreadable_openapi_is_warning(tmp_path):
    _, _, s, _ = setup(tmp_path)
    s.openapi_status = 500
    logs: list[str] = []
    pre(s, log=logs.append)
    assert any("openapi" in m for m in logs)


def test_preflight_schema_missing_is_warning(tmp_path):
    _, _, s, _ = setup(tmp_path)
    original = s.get

    def get(url, **k):
        r = original(url, **k)
        if url.endswith("/openapi.json"):
            return Resp(200, {"paths": {va.SAVE_ROUTE: {}}})
        return r

    s.get = get
    logs: list[str] = []
    pre(s, log=logs.append)
    assert any("data_type 지원을 확인하지 못했습니다" in m for m in logs)


def test_preflight_not_ready_and_unreachable(tmp_path):
    _, _, s, _ = setup(tmp_path)
    s.ready_status = 503
    with pytest.raises(InputError, match="503"):
        pre(s)
    s.ready_status = 200
    s.connect_error = True
    with pytest.raises(InputError, match="접속할 수 없습니다"):
        pre(s)


# ---------------------------------------------------------------- 호출


def save(s, records, **kw):
    params = dict(dataset_id="sif-x", subdir="law", chunk_file="05_chunks.jsonl", validation_file="06_validation.json", count=len(records))
    params.update(kw)
    return va.save_vectors(s, "http://127.0.0.1:8001", TOKEN, sleep=lambda x: None, **params)


def test_save_request_body_matches_agreed_call(tmp_path):
    _, _, s, records = setup(tmp_path)
    save(s, records)
    assert s.bodies[0] == {
        "dataset_id": "sif-x", "data_type": "accidents", "subdir": "law",
        "chunk_file": "05_chunks.jsonl", "validation_file": "06_validation.json",
    }


def test_save_default_timeout_is_long(tmp_path):
    _, _, s, records = setup(tmp_path)
    save(s, records)
    assert s.timeouts == [3600.0]


def test_save_without_validation_omits_field(tmp_path):
    _, _, s, records = setup(tmp_path)
    body = save(s, records, validation_file=None)
    assert "validation_file" not in s.bodies[0] and body["validation"] == "structural"


def test_save_retries_and_gives_up(tmp_path):
    _, _, s, records = setup(tmp_path)
    s.script = lambda route, n: 502 if n < 3 else None
    sleeps: list[float] = []
    va.save_vectors(s, "http://127.0.0.1:8001", TOKEN, dataset_id="sif-x", subdir="law", chunk_file="05_chunks.jsonl",
                    validation_file=None, count=6, sleep=sleeps.append)
    assert sleeps == [5.0, 10.0]
    s.attempts.clear()
    s.script = lambda route, n: 503
    with pytest.raises(IngestError, match="계속 실패"):
        save(s, records)


def test_save_422_and_404_not_retried(tmp_path):
    _, _, s, records = setup(tmp_path)
    s.script = lambda route, n: 422
    with pytest.raises(IngestError, match="거부되었습니다"):
        save(s, records)
    assert s.attempts["save"] == 1
    s.script = None
    with pytest.raises(IngestError, match=r"LOCAL_CHUNK_ROOT/accidents/other/"):
        save(s, records, subdir="other")


def test_save_wrong_token(tmp_path):
    _, _, s, records = setup(tmp_path)
    with pytest.raises(InputError, match="KB_INGEST_API_TOKEN"):
        va.save_vectors(s, "http://127.0.0.1:8001", "bad", dataset_id="x", subdir="law", chunk_file="05_chunks.jsonl",
                        validation_file=None, count=6)


def test_save_response_checks(tmp_path):
    _, _, s, records = setup(tmp_path)
    s.saved_delta = 1
    with pytest.raises(IngestError, match="청크 수가 다릅니다.*파일 6, 서버 7"):
        save(s, records)
    s.saved_delta = 0
    s.wrong_file = True
    with pytest.raises(IngestError, match="다른 파일에 저장"):
        save(s, records)
    s.wrong_file = False
    s.wrong_collection = True
    with pytest.raises(IngestError, match="kb_guides"):
        save(s, records)
    s.wrong_collection = False
    s.wrong_validation = True
    with pytest.raises(IngestError, match="검증 방식이 structural"):
        save(s, records)


def test_save_bool_saved_rejected(tmp_path):
    _, _, s, _ = setup(tmp_path, make_records(1))
    s.post = lambda *a, **k: Resp(200, {"saved": True, "dataset_id": "sif-x", "file": "sif-x.jsonl", "collection": "kb_accidents"})
    with pytest.raises(IngestError, match="청크 수가 다릅니다"):
        save(s, make_records(1))


# ---------------------------------------------------------------- 벡터 파일 검사


def made(tmp_path, records=None, **flags):
    chunks, vec, s, records = setup(tmp_path, records)
    for k, v in flags.items():
        setattr(s, k, v)
    save(s, records)
    return vec / "sif-x.jsonl", va.check_records(records), s


def rewrite(path, fn):
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    out = fn(lines)
    path.write_text("".join((json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x) + "\n" for x in out), encoding="utf-8")


def test_check_vector_file_ok(tmp_path):
    path, points, _ = made(tmp_path)
    assert va.check_vector_file(path, points, DIM) == (6, [])


def test_check_vector_file_missing_has_hint(tmp_path):
    with pytest.raises(IngestError, match="마운트.*--skip-file-check"):
        va.check_vector_file(tmp_path / "none.jsonl", va.check_records(make_records(2)), DIM)


def test_check_vector_file_dim_and_sparse(tmp_path):
    path, points, _ = made(tmp_path)
    _, problems = va.check_vector_file(path, points, DIM + 1)
    assert sum("차원이" in p for p in problems) == 6
    path2, points2, _ = made(tmp_path / "a", omit_sparse=True)
    assert sum("sparse" in p for p in va.check_vector_file(path2, points2, DIM)[1]) == 6
    path3, points3, _ = made(tmp_path / "b", empty_sparse=True)
    assert sum("sparse" in p for p in va.check_vector_file(path3, points3, DIM)[1]) == 6


def test_check_vector_file_missing_extra_dup(tmp_path):
    path, points, _ = made(tmp_path)
    rewrite(path, lambda L: L[:-2] + [L[0]])
    _, problems = va.check_vector_file(path, points, DIM)
    assert any("중복 id" in p for p in problems) and any("벡터 파일에 없는 청크 2건" in p for p in problems)
    path, points, _ = made(tmp_path / "c")
    rewrite(path, lambda L: L + [{"id": "f" * 8 + "-0000-0000-0000-000000000000", "vector": {}, "payload": {}}])
    assert any("기대하지 않은 id" in p for p in va.check_vector_file(path, points, DIM)[1])


@pytest.mark.parametrize("field,value", [
    ("text", "옛 개정 본문"), ("title", "다른 제목"), ("accident_type", ["STRUCK"]), ("hazard", ["FIRE"]),
    ("severity", None), ("year", 2020), ("site_id", 3), ("flagged", True), ("source_type", "LAW"), ("locator", "x"),
    ("external_id", "SIF-other"), ("chunk_index", 5),
])
def test_check_vector_file_detects_payload_drift(tmp_path, field, value):
    path, points, _ = made(tmp_path)

    def drift(lines):
        lines[1]["payload"][field] = value
        return lines

    rewrite(path, drift)
    _, problems = va.check_vector_file(path, points, DIM)
    assert any("payload가 05와 다름" in p and field in p for p in problems)


def test_check_vector_file_type_drift_detected(tmp_path):
    path, points, _ = made(tmp_path)

    def drift(lines):
        lines[0]["payload"]["year"] = 2019.0  # 값은 같지만 형이 다르다
        lines[2]["payload"]["flagged"] = 0
        return lines

    rewrite(path, drift)
    _, problems = va.check_vector_file(path, points, DIM)
    assert any("year" in p for p in problems) and any("flagged" in p for p in problems)


def test_check_vector_file_linked_ids_flagged(tmp_path):
    path, points, _ = made(tmp_path)

    def link(lines):
        lines[0]["payload"]["kb_source_id"] = 7
        return lines

    rewrite(path, link)
    assert any("kb_source_id가 null이 아님" in p for p in va.check_vector_file(path, points, DIM)[1])


def test_check_vector_file_malformed(tmp_path):
    path, points, _ = made(tmp_path)
    rewrite(path, lambda L: ["not json", {"id": "x"}] + L)
    _, problems = va.check_vector_file(path, points, DIM)
    assert any("JSON이 아닙니다" in p for p in problems) and any("id·vector·payload가 필요" in p for p in problems)


def test_check_vector_file_bad_payload_and_vector(tmp_path):
    path, points, _ = made(tmp_path)

    def bad(lines):
        lines[0]["payload"] = "x"
        lines[1]["vector"]["dense"][0] = True
        lines[2]["vector"]["sparse"] = {"indices": [1], "values": [1.0, 2.0]}
        return lines

    rewrite(path, bad)
    _, problems = va.check_vector_file(path, points, DIM)
    assert any("payload가 객체가 아님" in p for p in problems)
    assert any("dense" in p for p in problems) and any("sparse" in p for p in problems)


def test_check_vector_file_ignores_blank_lines(tmp_path):
    path, points, _ = made(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert va.check_vector_file(path, points, DIM)[1] == []


def test_file_digest(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert va.file_digest(p) == (3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# ---------------------------------------------------------------- 인자


def test_parse_args_defaults():
    args = va.parse_args([])
    assert args.chunks == Path("out/accidents/law/05_chunks.jsonl") and args.subdir == "law"
    assert args.vector_dir == Path("out/vectors") and args.embedding_dim == 1024 and args.save_timeout == 3600.0
    assert args.limit is None and not args.dry_run


@pytest.mark.parametrize("argv", [["--embedding-dim", "0"], ["--save-timeout", "0"], ["--limit", "0"], ["--check-only", "--skip-file-check"]])
def test_parse_args_rejects(argv):
    with pytest.raises(SystemExit):
        va.parse_args(argv)


def test_parse_args_env_defaults(monkeypatch):
    monkeypatch.setenv("KB_INGEST_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("EMBEDDING_DIM", "512")
    args = va.parse_args([])
    assert (args.ingest_url, args.embedding_dim) == ("http://127.0.0.1:9", 512)


# ---------------------------------------------------------------- 전체 실행


def test_run_dry_run_touches_nothing(tmp_path):
    chunks, vec, _, records = setup(tmp_path)
    logs: list[str] = []
    assert va.run(args_for(chunks, vec, "--dry-run"), session=Boom(), log=logs.append, clock=lambda: NOW) == 0
    text = "\n".join(logs)
    assert "SIF 청크 6건" in text and "sif-2026-11.jsonl" in text and "pipeline" in text and "[dry-run]" in text
    assert "<LOCAL_CHUNK_ROOT>/accidents/law/05_chunks.jsonl" in text
    assert not vec.exists()


def test_run_dry_run_needs_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("KB_INGEST_API_TOKEN")
    chunks, vec, _, _ = setup(tmp_path)
    assert va.run(args_for(chunks, vec, "--dry-run"), session=Boom(), log=lambda *_: None, clock=lambda: NOW) == 0


def test_run_full(tmp_path):
    chunks, vec, s, records = setup(tmp_path)
    logs: list[str] = []
    assert go(args_for(chunks, vec), s, logs) == 0
    text = "\n".join(logs)
    assert "[완료] 벡터 파일 검사 통과: 6점" in text and "SHA256:" in text and "별도 적재 스크립트" in text
    assert (vec / "sif-2026-11.jsonl").is_file()
    assert s.bodies[0]["validation_file"] == "06_validation.json" and s.bodies[0]["data_type"] == "accidents"
    # Qdrant·장부·다른 경로는 건드리지 않는다
    assert [url for kind, url in s.calls if kind == "POST"] == ["http://127.0.0.1:8001/ingest/vectors/local/file"]


def test_run_passes_save_timeout(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    go(args_for(chunks, vec), s)
    go(args_for(chunks, vec, "--save-timeout", "42"), s)
    assert s.timeouts == [3600.0, 42.0]


def test_run_never_calls_load_or_document_or_chunks(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    go(args_for(chunks, vec), s)
    assert not any(x in url for _, url in s.calls for x in ("/ingest/document", "/ingest/chunks", "/load", "qdrant", ":60011", "points"))


def test_run_without_validation_file_present(tmp_path):
    chunks, vec, s, _ = setup(tmp_path, validation=None)
    logs: list[str] = []
    assert go(args_for(chunks, vec), s, logs) == 0
    assert "validation_file" not in s.bodies[0] and any("구조 검사만" in m for m in logs)


def test_run_no_validation_flag(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    assert go(args_for(chunks, vec, "--no-validation"), s) == 0
    assert "validation_file" not in s.bodies[0]


def test_run_bad_validation_stops_before_any_call(tmp_path):
    chunks, vec, s, _ = setup(tmp_path, validation={"ok": False, "dataset": "sif", "summary": {"chunks": 6}})
    with pytest.raises(InputError, match="ok가 true가 아닙니다"):
        go(args_for(chunks, vec), s)
    assert s.calls == []


def test_run_bad_record_stops_before_any_call(tmp_path):
    records = make_records(3)
    records[1]["flagged"] = "no"
    chunks, vec, s, _ = setup(tmp_path, records)
    with pytest.raises(InputError, match="flagged"):
        go(args_for(chunks, vec), s)
    assert s.calls == []


def test_run_limit_writes_partial_file_and_skips_validation(tmp_path):
    chunks, vec, s, records = setup(tmp_path, make_records(10))
    assert go(args_for(chunks, vec, "--limit", "4"), s) == 0
    part = chunks.with_name("05_chunks.limit4.jsonl")
    assert part.is_file() and len(part.read_text(encoding="utf-8").splitlines()) == 4
    assert s.bodies[0]["chunk_file"] == "05_chunks.limit4.jsonl" and s.bodies[0]["dataset_id"] == "sif-limit4"
    assert "validation_file" not in s.bodies[0]
    assert len((vec / "sif-limit4.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    assert len(chunks.read_text(encoding="utf-8").splitlines()) == 10  # 원본은 그대로


def test_run_limit_larger_than_file(tmp_path):
    chunks, vec, s, _ = setup(tmp_path, make_records(3))
    assert go(args_for(chunks, vec, "--limit", "50"), s) == 0
    assert s.bodies[0]["chunk_file"] == "05_chunks.limit3.jsonl"


def test_run_limit_dry_run_writes_nothing(tmp_path):
    chunks, vec, _, _ = setup(tmp_path, make_records(10))
    assert va.run(args_for(chunks, vec, "--limit", "4", "--dry-run"), session=Boom(), log=lambda *_: None, clock=lambda: NOW) == 0
    assert not chunks.with_name("05_chunks.limit4.jsonl").exists()


def test_run_limit_still_checks_whole_file(tmp_path):
    records = make_records(10)
    records[8]["title"] = ""
    chunks, vec, s, _ = setup(tmp_path, records)
    with pytest.raises(InputError, match="title 값이 비어"):
        go(args_for(chunks, vec, "--limit", "4"), s)


def test_run_custom_dataset_and_subdir(tmp_path):
    chunks, vec, s, records = setup(tmp_path)
    other = tmp_path / "out" / "accidents" / "sif20261110"
    write_chunks(other, records)
    s.subdir = "sif20261110"
    assert go(args_for(chunks, vec, "--subdir", "sif20261110", "--dataset-id", "sif-test-1"), s) == 0
    assert s.bodies[0]["subdir"] == "sif20261110" and (vec / "sif-test-1.jsonl").is_file()


def test_run_bad_inputs():
    with pytest.raises(InputError, match="--subdir"):
        va.run(va.parse_args(["--subdir", "../x", "--dry-run"]), session=Boom(), log=lambda *_: None)
    with pytest.raises(InputError, match="청크 파일 이름"):
        va.run(va.parse_args(["--chunks", "out/청크.jsonl", "--dry-run"]), session=Boom(), log=lambda *_: None)


def test_run_bad_dataset_id(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    with pytest.raises(InputError, match="dataset_id"):
        go(args_for(chunks, vec, "--dataset-id", "bad id"), s)


def test_run_missing_chunk_file(tmp_path):
    with pytest.raises(InputError, match="청크 파일이 없습니다"):
        va.run(va.parse_args(["--chunks", str(tmp_path / "x.jsonl"), "--dry-run"]), session=Boom(), log=lambda *_: None)


def test_run_missing_token(tmp_path, monkeypatch):
    monkeypatch.delenv("KB_INGEST_API_TOKEN")
    chunks, vec, s, _ = setup(tmp_path)
    with pytest.raises(InputError, match="KB_INGEST_API_TOKEN"):
        go(args_for(chunks, vec), s)
    assert s.calls == []


def test_run_remote_refused_without_flag(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    with pytest.raises(InputError, match="로컬이 아닙니다"):
        go(args_for(chunks, vec, "--ingest-url", "http://10.1.1.1:8001"), s)
    assert s.calls == []
    assert go(args_for(chunks, vec, "--ingest-url", "http://10.1.1.1:8001", "--allow-remote"), s) == 0


def test_run_old_server_stops_before_post(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    s.schema_has_data_type = False
    with pytest.raises(InputError, match="옛 판"):
        go(args_for(chunks, vec), s)
    assert s.attempts["save"] == 0


def test_run_server_down(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    s.connect_error = True
    with pytest.raises(InputError, match="접속할 수 없습니다"):
        go(args_for(chunks, vec), s)


def test_run_stale_server_copy_detected(tmp_path):
    """서버가 읽는 사본이 스크립트가 검사한 파일과 다르면 벡터 파일 검사에서 멈춘다."""
    chunks, vec, s, records = setup(tmp_path)
    server_root = tmp_path / "server_root"
    stale = [dict(r, text=r["text"] + " (옛 판)") for r in records]
    write_chunks(server_root / "accidents" / "law", stale)
    s.root = server_root
    with pytest.raises(IngestError, match="payload가 05와 다름"):
        go(args_for(chunks, vec), s)


def test_run_server_copy_with_other_tags_detected(tmp_path):
    chunks, vec, s, records = setup(tmp_path)
    server_root = tmp_path / "server_root"
    other = [dict(r, accident_type=["STRUCK"]) for r in records]
    write_chunks(server_root / "accidents" / "law", other)
    s.root = server_root
    with pytest.raises(IngestError, match="accident_type"):
        go(args_for(chunks, vec), s)


def test_run_server_copy_with_other_ids_detected(tmp_path):
    chunks, vec, s, records = setup(tmp_path)
    server_root = tmp_path / "server_root"
    write_chunks(server_root / "accidents" / "law", make_records(7)[1:])
    s.root = server_root
    with pytest.raises(IngestError, match="기대하지 않은 id|payload가 05와 다름"):
        go(args_for(chunks, vec), s)


def test_run_old_embedding_without_sparse(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    s.omit_sparse = True
    with pytest.raises(IngestError, match="sparse"):
        go(args_for(chunks, vec), s)


def test_run_check_only_after_save(tmp_path, monkeypatch):
    chunks, vec, s, _ = setup(tmp_path)
    go(args_for(chunks, vec), s)
    s.calls.clear()
    monkeypatch.delenv("KB_INGEST_API_TOKEN")
    assert go(args_for(chunks, vec, "--check-only"), s) == 0
    assert s.calls == []  # API 호출 없음


def test_run_check_only_without_file_explains(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    with pytest.raises(IngestError, match="호스트에 없습니다"):
        go(args_for(chunks, vec, "--check-only"), s)


def test_run_check_only_detects_tampered_file(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    go(args_for(chunks, vec), s)
    rewrite(vec / "sif-2026-11.jsonl", lambda L: L[:-1])
    with pytest.raises(IngestError, match="벡터 파일에 없는 청크 1건"):
        go(args_for(chunks, vec, "--check-only"), s)


def test_run_skip_file_check_when_not_mounted(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    s.no_write = True
    logs: list[str] = []
    assert go(args_for(chunks, vec, "--skip-file-check"), s, logs) == 0
    assert any("검사를 건너뛰었습니다" in m for m in logs)


def test_run_rerun_overwrites_identically(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    go(args_for(chunks, vec), s)
    first = (vec / "sif-2026-11.jsonl").read_bytes()
    go(args_for(chunks, vec), s)
    assert (vec / "sif-2026-11.jsonl").read_bytes() == first


def test_run_does_not_modify_inputs(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    before = {p.name: p.read_bytes() for p in chunks.parent.iterdir()}
    go(args_for(chunks, vec), s)
    assert {p.name: p.read_bytes() for p in chunks.parent.iterdir()} == before


def test_run_reports_digest_of_written_file(tmp_path):
    chunks, vec, s, _ = setup(tmp_path)
    logs: list[str] = []
    go(args_for(chunks, vec), s, logs)
    size, digest = va.file_digest(vec / "sif-2026-11.jsonl")
    assert any(digest in m for m in logs) and any(f"{size:,}" in m for m in logs)


# ---------------------------------------------------------------- main 종료 코드


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    chunks, _, _, _ = setup(tmp_path)
    monkeypatch.setattr(va, "run", lambda args: 0)
    with pytest.raises(SystemExit) as ok:
        va.main(["--chunks", str(chunks), "--dry-run"])
    assert ok.value.code == 0
    monkeypatch.setattr(va, "run", lambda args: (_ for _ in ()).throw(InputError("입력")))
    with pytest.raises(SystemExit) as bad:
        va.main(["--chunks", str(chunks)])
    assert bad.value.code == 2 and "[입력 오류] 입력" in capsys.readouterr().err
    monkeypatch.setattr(va, "run", lambda args: (_ for _ in ()).throw(IngestError("중단")))
    with pytest.raises(SystemExit) as failed:
        va.main(["--chunks", str(chunks)])
    assert failed.value.code == 1 and "[중단] 중단" in capsys.readouterr().err
    monkeypatch.setattr(va, "run", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(SystemExit) as stopped:
        va.main(["--chunks", str(chunks)])
    assert stopped.value.code == 1 and "덮어씁니다" in capsys.readouterr().err


def test_source_hygiene():
    source = Path(va.__file__).read_text(encoding="utf-8")
    assert "from app" not in source and "import app" not in source
    assert "pymysql" not in source and "qdrant" not in source.lower().replace("qdrant에는", "").replace("qdrant 올리기", "").replace("qdrant·", "").replace("(qdrant", "").replace("scripts/upsert_qdrant_jsonl.py", "").replace("upsert_qdrant", "")
    assert "\r" not in source
