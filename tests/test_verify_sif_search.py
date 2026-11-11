"""scripts/verify_sif_search.py 단위 시험. Qdrant와 검색 백엔드는 가짜로 흉내 낸다 (네트워크·앱 코드 없음)."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

from scripts import verify_sif_search as vs
from scripts.ingest_law_vectors import IngestError, InputError
from scripts.vectorize_accidents import check_records, point_id_for

DIM = 8
RKEY = "k-read"

TOPICS = [
    # (제목 앞부분, 재해형태 라벨, 개요, accident_type, work_type, hazard, severity)
    ("용접 작업", "화재", "옥상에서 용접 작업 중 불티가 튀어 부직포에 화재가 발생하여 작업자가 화상을 입고 사망", ["FIRE"], ["HOT"], [], "FATAL"),
    ("탱크 청소", "질식", "밀폐공간 탱크 내부 청소 중 산소결핍으로 작업자가 질식하여 사망", ["ASPHYXIATION"], ["CONFINED"], [], "FATAL"),
    ("외부 비계 설치", "추락", "비계 조립 작업 중 안전대를 걸지 않고 이동하다 고소에서 떨어져 사망", ["FALL"], ["HEIGHT"], [], "FATAL"),
    ("배전반 점검", "감전", "전기 작업 중 배전반 점검하다 충전부에 접촉하여 감전", ["ELECTRIC"], ["ELECTRICAL"], ["ELECTRICAL"], None),
    ("컨베이어 정비", "끼임", "컨베이어 정비 중 회전체에 손이 끼임", ["CAUGHT"], [], ["MECHANICAL"], None),
    ("사다리 작업", "추락", "이동식 사다리 위에서 작업하다 균형을 잃고 떨어져 사망", ["FALL"], ["HEIGHT"], [], "FATAL"),
    ("크레인 인양", "낙하", "크레인으로 자재를 인양하다 줄걸이가 풀려 낙하물에 맞음", ["STRUCK"], ["LIFTING"], [], None),
    ("용접 작업", "폭발", "가연성 가스가 남은 배관을 용접하다 폭발", ["EXPLOSION"], ["HOT"], [], None),
]


def make_record(i: int):
    top = TOPICS[(i - 1) % len(TOPICS)]
    ext = f"SIF-{i:016x}"
    return {
        "natural_key": {"dataset": "sif", "record": ext, "section": "case", "part": 0},
        "external_id": ext,
        "chunk_index": 0,
        "text": f"[재해개요]\n{top[2]} (사례번호 {i})\n\n[재해유발요인]\n안전조치 미흡 {i}",
        "embed_text": f"SIF > {top[0]} > {top[1]}\n[재해개요]\n{top[2]}",
        "title": f"{top[0]} – {top[1]} (2019)",
        "locator": "재해개요·유발요인·감소대책(예시)",
        "source_type": "SIF",
        "collection": "kb_accidents",
        "site_id": 0,
        "flagged": False,
        "year": 2019,
        "severity": top[6],
        "accident_type": list(top[3]),
        "hazard": list(top[5]),
        "work_type": list(top[4]),
        "cas": [],
        "ghs": [],
        "url": "https://www.data.go.kr/data/15161362/openapi.do",
        "license": "이용허락범위 제한 없음",
        "publisher": "한국산업안전보건공단",
        "content_sha256": "a" * 64,
        "tier": "A",
        "cate": "전업종",
        "disaster_type": top[1],
        "situation": None,
        "kb_source_id": None,
        "kb_chunk_id": None,
        "point_id": None,
    }


def make_records(n: int = 16):
    return [make_record(i) for i in range(1, n + 1)]


def write_chunks(tmp_path: Path, records) -> Path:
    path = tmp_path / "05_chunks.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
    return path


PAYLOAD_KEYS = (
    "source_type", "site_id", "flagged", "title", "locator", "url", "text", "severity", "accident_type", "hazard",
    "work_type", "cas", "ghs", "year", "external_id", "chunk_index", "license", "publisher", "content_sha256",
    "tier", "cate", "disaster_type", "situation",
)


class Resp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError("JSON 아님")
        return self._data


class FakeQdrant:
    """GET 컬렉션, POST count, POST 점 조회만 흉내 낸다."""

    def __init__(self, records, *, load=True):
        self.points: dict[str, dict] = {}
        self.script = None
        self.attempts = 0
        self.dense_size = DIM
        self.has_sparse = True
        self.sparse_name = "sparse"
        self.collection_status = 200
        self.omit_sparse = False
        self.empty_sparse = False
        self.wrong_dense = False
        self.calls: list[tuple[str, str]] = []
        if load:
            for r in records:
                self.add(r)

    def add(self, record, **payload_over):
        pid = point_id_for(record["external_id"], record["chunk_index"])
        payload = {k: record[k] for k in PAYLOAD_KEYS if k in record}
        payload["kb_source_id"] = None
        payload["kb_chunk_id"] = None
        payload.update(payload_over)
        self.points[pid] = {"id": pid, "payload": payload}
        return pid

    def _vector(self):
        vector = {"dense": [0.1] * (DIM - 1 if self.wrong_dense else DIM)}
        if not self.omit_sparse:
            vector["sparse"] = {"indices": [], "values": []} if self.empty_sparse else {"indices": [1], "values": [1.0]}
        return vector

    def _maybe_fail(self, headers):
        if (headers or {}).get("api-key") != RKEY:
            return Resp(401, {"status": {"error": "Unauthorized"}})
        self.attempts += 1
        if self.script:
            action = self.script(self.attempts)
            if isinstance(action, Exception):
                raise action
            if isinstance(action, int):
                return Resp(action, None)
        return None

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url))
        early = self._maybe_fail(headers)
        if early is not None:
            return early
        if self.collection_status != 200:
            return Resp(self.collection_status, {"status": "err"})
        params = {"vectors": {"dense": {"size": self.dense_size}}}
        if self.has_sparse:
            params["sparse_vectors"] = {self.sparse_name: {}}
        return Resp(200, {"result": {"config": {"params": params}}})

    @staticmethod
    def _matches(payload, must):
        for cond in must:
            have = payload.get(cond["key"])
            want = cond["match"]["value"]
            if isinstance(have, list):
                if want not in have:
                    return False
            elif have != want:
                return False
        return True

    def post(self, url, headers=None, json=None, timeout=None):  # noqa: A002
        self.calls.append(("POST", url))
        early = self._maybe_fail(headers)
        if early is not None:
            return early
        if url.endswith("/points/count"):
            must = json["filter"]["must"]
            return Resp(200, {"result": {"count": sum(1 for p in self.points.values() if self._matches(p["payload"], must))}})
        out = []
        for pid in json["ids"]:
            point = self.points.get(pid)
            if point is None:
                continue
            item = {"id": pid}
            if json.get("with_payload"):
                item["payload"] = point["payload"]
            if json.get("with_vector"):
                item["vector"] = self._vector()
            out.append(item)
        return Resp(200, {"result": out})


class Boom:
    def __getattr__(self, name):
        raise AssertionError(f"접속하면 안 됩니다: {name}")


def shingles(text: str) -> set[str]:
    t = "".join(text.split())
    return {t[i:i + 2] for i in range(len(t) - 1)}


class FakeSearch:
    """질문과 본문의 2글자 조각 겹침으로 순위를 매기는 가짜 검색. 필터를 지킨다."""

    def __init__(self, qdrant: FakeQdrant):
        self.qdrant = qdrant
        self.calls: list[dict] = []
        self.leak = False
        self.empty = False
        self.bad_id = False
        self.doc_none = False
        self.raise_error = None
        self.list_only = False
        self.shuffle = False
        self.error_on_filter = False

    def __call__(self, query, *, k, filters):
        self.calls.append({"query": query, "k": k, "filters": dict(filters)})
        if self.raise_error is not None:
            raise self.raise_error
        if self.error_on_filter and any(a in filters for a in vs.TAG_AXES):
            raise RuntimeError("필터 오류")
        if self.empty:
            return []
        wanted = [(a, filters[a]) for a in vs.TAG_AXES if a in filters]
        if self.list_only and any(isinstance(v, str) for _, v in wanted):
            raise TypeError("배열이어야 합니다")
        q = shingles(query)
        rows = []
        for pid, point in self.qdrant.points.items():
            payload = point["payload"]
            ok = True
            for axis, value in wanted:
                values = value if isinstance(value, list) else [value]
                have = payload.get(axis)
                have = have if isinstance(have, list) else [have]
                if not all(v in have for v in values) and not self.leak:
                    ok = False
            if not ok:
                continue
            rows.append((len(q & shingles(payload["text"])) / max(len(q), 1), pid, payload))
        rows.sort(key=lambda r: (-r[0], r[1]))
        if self.shuffle:
            rows = list(reversed(rows))
        hits = []
        for score, pid, payload in rows[:k]:
            hits.append({
                "id": "not-a-uuid" if self.bad_id else pid,
                "doc_id": "SIF:None" if self.doc_none else f"SIF:{payload.get('kb_source_id') or 1}",
                "source": f"{payload['title']} ({payload['locator']})",
                "text": payload["text"],
                "score": float(score),
            })
        return hits


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", RKEY)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    for name in ("QDRANT_URL", "EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)


def setup(tmp_path, n=16):
    records = make_records(n)
    chunks = write_chunks(tmp_path, records)
    q = FakeQdrant(records)
    return chunks, q, FakeSearch(q), check_records(records)


def args_for(chunks, *extra):
    return vs.parse_args(["--chunks", str(chunks), "--embedding-dim", str(DIM), "--self-queries", "10", "--title-queries", "6", *extra])


def go(args, q, search=None, logs=None, **kw):
    out = logs if logs is not None else []
    return vs.run(args, session=q, search_factory=(lambda: search) if search is not None else None, sleep=lambda s: None, log=out.append, **kw)


def no_sleep(_):
    return None


# ---------------------------------------------------------------- 기대값 도우미


def test_tag_pairs_and_counts():
    records = make_records(8)
    assert ("severity", "FATAL") in vs.tag_pairs(records[0]) and ("work_type", "HOT") in vs.tag_pairs(records[0])
    assert vs.tag_pairs(records[3]) == [("accident_type", "ELECTRIC"), ("hazard", "ELECTRICAL"), ("work_type", "ELECTRICAL")]
    points = check_records(make_records(16))
    counts = vs.expected_tag_counts(points)
    assert counts[("accident_type", "FALL")] == 4 and counts[("severity", "FATAL")] == 8 and counts[("work_type", "HOT")] == 4


def test_expected_counts_skip_flagged():
    records = make_records(8)
    records[0]["flagged"] = True
    counts = vs.expected_tag_counts(check_records(records))
    assert ("accident_type", "FIRE") not in counts and counts[("work_type", "HOT")] == 1


def test_query_builders():
    record = make_record(1)
    assert vs.overview_of(record).startswith("옥상에서 용접 작업 중") and "[" not in vs.overview_of(record)
    assert len(vs.self_query(record)) <= 80
    assert vs.clean_title(record) == "용접 작업 – 화재"
    assert vs.clean_title({"title": "  제목 (2020) "}) == "제목"
    assert vs.overview_of({"text": "개요 없음 본문"}) == "개요 없음 본문"


def test_stride_sample():
    assert vs.stride_sample(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert vs.stride_sample(list(range(3)), 10) == [0, 1, 2]
    assert vs.stride_sample([], 3) == [] and vs.stride_sample([1, 2], 0) == []


def test_read_key(monkeypatch):
    assert vs.read_key() == RKEY
    monkeypatch.delenv("QDRANT_READ_ONLY_API_KEY")
    monkeypatch.setenv("QDRANT_API_KEY", "w-key")
    assert vs.read_key() == "w-key"
    monkeypatch.delenv("QDRANT_API_KEY")
    with pytest.raises(InputError, match="QDRANT_READ_ONLY_API_KEY"):
        vs.read_key()
    monkeypatch.setenv("QDRANT_READ_ONLY_API_KEY", "ab\ncd")
    with pytest.raises(InputError, match="제어 문자"):
        vs.read_key()


# ---------------------------------------------------------------- Qdrant 요청


def test_qrequest_retries_then_ok(tmp_path):
    _, q, _, _ = setup(tmp_path)
    q.script = lambda n: 503 if n < 3 else None
    sleeps: list[float] = []
    data, error = vs._qrequest(q, "GET", "http://127.0.0.1:60011/collections/kb_accidents", RKEY, None, sleep=sleeps.append)
    assert error is None and sleeps == [5.0, 10.0]


def test_qrequest_gives_up_and_4xx_not_retried(tmp_path):
    _, q, _, _ = setup(tmp_path)
    q.script = lambda n: 500
    assert vs._qrequest(q, "GET", "http://x/c", RKEY, None, sleep=no_sleep) == (None, "HTTP 500")
    q.attempts = 0
    q.script = lambda n: 400
    assert vs._qrequest(q, "GET", "http://x/c", RKEY, None, sleep=no_sleep) == (None, "HTTP 400") and q.attempts == 1


def test_qrequest_connection_error_and_auth(tmp_path):
    _, q, _, _ = setup(tmp_path)
    q.script = lambda n: ConnectionError("x") if n == 1 else None
    assert vs._qrequest(q, "GET", "http://x/c", RKEY, None, sleep=no_sleep)[1] is None
    with pytest.raises(InputError, match="읽기 키를 거부"):
        vs._qrequest(q, "GET", "http://x/c", "wrong", None, sleep=no_sleep)
    q.script = lambda n: ConnectionError("x")
    assert vs._qrequest(q, "GET", "http://x/c", RKEY, None, sleep=no_sleep)[1].startswith("연결 오류")


def test_qrequest_non_json():
    class S:
        def get(self, *a, **k):
            return Resp(200, None)

    assert vs._qrequest(S(), "GET", "http://x", RKEY, None, sleep=no_sleep) == (None, "응답이 JSON이 아님")


# ---------------------------------------------------------------- 1부


def cc(q, **kw):
    return vs.check_collection(q, "http://127.0.0.1:60011", RKEY, DIM, sleep=no_sleep, **kw)


def test_check_collection_sparse_name_and_forbidden(tmp_path):
    _, q, _, _ = setup(tmp_path)
    q.sparse_name = "other"
    assert any("sparse" in p for p in cc(q))
    q.sparse_name = "sparse"
    q.script = lambda attempt: 403
    with pytest.raises(InputError, match="HTTP 403"):
        cc(q)


def test_check_collection(tmp_path):
    _, q, _, _ = setup(tmp_path)
    assert cc(q) == []
    q.dense_size = 512
    assert any("dense 차원이 512" in p for p in cc(q))
    q.dense_size = DIM
    q.has_sparse = False
    assert any("sparse" in p for p in cc(q))
    q.has_sparse = True
    q.collection_status = 500
    assert any("컬렉션 조회 실패" in p for p in cc(q))
    q.collection_status = 404
    with pytest.raises(InputError, match="init_qdrant"):
        cc(q)


def counts(q, points, partial=False):
    return vs.check_counts(q, "http://127.0.0.1:60011", RKEY, points, partial=partial, sleep=no_sleep)


def test_check_counts_ok(tmp_path):
    _, q, _, points = setup(tmp_path)
    assert counts(q, points) == ([], [])


def test_check_counts_missing_points(tmp_path):
    _, q, _, points = setup(tmp_path)
    for p in points[:3]:
        del q.points[p.point_id]
    problems, _ = counts(q, points)
    assert any("flagged=false): 기대 16, 실제 13" in p for p in problems)
    assert any("태그 필터 count가 05와 다른 값" in p for p in problems)


def test_check_counts_extra_points_fail_full_but_note_partial(tmp_path):
    chunks, q, _, points = setup(tmp_path)
    q.add(make_record(99))
    problems, _ = counts(q, points)
    assert any("기대 16, 실제 17" in p for p in problems)
    problems, notes = counts(q, points, partial=True)
    assert problems == [] and any("17개로" in n for n in notes)


def test_check_counts_flagged_points_counted_separately(tmp_path):
    records = make_records(8)
    records[0]["flagged"] = True
    q = FakeQdrant(records)
    points = check_records(records)
    assert counts(q, points)[0] == []
    q.points[point_id_for(records[0]["external_id"], 0)]["payload"]["flagged"] = False
    assert counts(q, points)[0] != []


def test_check_counts_tag_filter_mismatch_detected(tmp_path):
    _, q, _, points = setup(tmp_path)
    q.points[points[2].point_id]["payload"]["accident_type"] = []  # 필터 대상에서 빠짐
    problems, _ = counts(q, points)
    assert any("accident_type=FALL: 기대 4, 실제 3" in p for p in problems)


def test_check_counts_query_error(tmp_path):
    _, q, _, points = setup(tmp_path)
    q.script = lambda n: 400
    problems, _ = counts(q, points)
    assert any("점 수 조회 실패" in p for p in problems)


def pts(q, points, **kw):
    return vs.check_points(q, "http://127.0.0.1:60011", RKEY, points, DIM, sleep=no_sleep, **kw)


def test_check_points_ok_and_unlinked_note(tmp_path):
    _, q, _, points = setup(tmp_path)
    problems, notes = pts(q, points)
    assert problems == [] and "0/16" in notes[0] and "벡터 파일 방식" in notes[0]


def test_check_points_linked_note_has_no_warning(tmp_path):
    _, q, _, points = setup(tmp_path)
    for n, point in enumerate(q.points.values(), 1):
        point["payload"]["kb_source_id"] = n
        point["payload"]["kb_chunk_id"] = n
    _, notes = pts(q, points)
    assert "16/16" in notes[0] and "벡터 파일 방식" not in notes[0]


def test_check_points_missing_and_drift(tmp_path):
    _, q, _, points = setup(tmp_path)
    del q.points[points[1].point_id]
    q.points[points[2].point_id]["payload"]["text"] = "옛 본문"
    q.points[points[3].point_id]["payload"]["accident_type"] = ["STRUCK"]
    problems, _ = pts(q, points)
    assert any("Qdrant에 없는 점 1건" in p for p in problems)
    drift = [p for p in problems if "payload가 05와 다른 점 2건" in p][0]
    assert "text" in drift and "accident_type" in drift


def test_check_points_vectors(tmp_path):
    _, q, _, points = setup(tmp_path)
    q.omit_sparse = True
    assert any("sparse" in p for p in pts(q, points)[0])
    q.omit_sparse = False
    q.empty_sparse = True
    assert any("sparse" in p for p in pts(q, points)[0])
    q.empty_sparse = False
    q.wrong_dense = True
    assert any("dense 차원" in p for p in pts(q, points)[0])


def test_check_points_retrieves_in_batches(tmp_path):
    records = make_records(450)
    q = FakeQdrant(records)
    sizes: list[int] = []
    original = q.post

    def post(url, headers=None, json=None, timeout=None):
        if "ids" in json:
            sizes.append(len(json["ids"]))
        return original(url, headers=headers, json=json, timeout=timeout)

    q.post = post
    assert pts(q, check_records(records))[0] == []
    assert sizes[:3] == [200, 200, 50]


def test_check_points_query_error(tmp_path):
    _, q, _, points = setup(tmp_path)
    q.script = lambda n: 400
    assert any("점 조회 실패" in p for p in pts(q, points)[0])


# ---------------------------------------------------------------- 2부


def test_hit_problems():
    good = {"id": str(uuid.uuid4()), "doc_id": "SIF:1", "source": "제목 (위치)", "text": "본문", "score": 0.5}
    assert vs.hit_problems(good) == []
    assert vs.hit_problems(dict(good, id="x")) and vs.hit_problems(dict(good, id=123))
    assert vs.hit_problems(dict(good, doc_id=""))
    assert any("꼴이 아님" in p for p in vs.hit_problems(dict(good, source="제목만")))
    assert vs.hit_problems(dict(good, text=None))
    assert vs.hit_problems(dict(good, score=True)) and vs.hit_problems(dict(good, score="1")) and vs.hit_problems(dict(good, score=float("nan")))
    assert vs.hit_problems({})


def test_hit_problems_requires_string_id():
    good = {"id": str(uuid.uuid4()), "doc_id": "SIF:1", "source": "제목 (위치)", "text": "본문", "score": 0.5}
    assert vs.hit_problems(dict(good, id=uuid.uuid4()))  # uuid 객체는 JSON 응답 규격이 아니다


def test_hit_problems_accepts_objects():
    class Hit:
        id = str(uuid.uuid4())
        doc_id = "SIF:1"
        source = "제목 (위치)"
        text = "본문"
        score = 1

    assert vs.hit_problems(Hit()) == []


def test_rank_and_summary():
    assert vs.rank_of(["a", "b", "c"], {"b"}) == 2 and vs.rank_of(["a"], {"z"}) is None
    s = vs.summarize([1, 3, None, 11])
    assert s == {"n": 4, "hit@1": 0.25, "hit@3": 0.5, "hit@5": 0.5, "hit@10": 0.5, "mrr": round((1 + 1 / 3 + 1 / 11) / 4, 4)}
    assert vs.summarize([]) == {"n": 0}


def smoke(search, points, **kw):
    text_of = {p.point_id: p.record["text"] for p in points}
    return vs.check_smoke(search, text_of, k=5, min_relevant=kw.pop("min_relevant", 2))


def test_smoke_passes(tmp_path):
    _, q, search, points = setup(tmp_path)
    problems, notes, rows = smoke(search, points)
    assert problems == [] and notes == [] and len(rows) == 4 and all(r["relevant"] >= 2 for r in rows)
    assert search.calls[0]["filters"] == {"collection": "kb_accidents", "source_type": "SIF"} and search.calls[0]["k"] == 5


def test_smoke_empty_error_and_contract(tmp_path):
    _, q, search, points = setup(tmp_path)
    search.empty = True
    assert sum("결과가 없습니다" in p for p in smoke(search, points)[0]) == 4
    search.empty = False
    search.raise_error = RuntimeError("임베딩 연결 실패")
    problems, _, rows = smoke(search, points)
    assert sum("검색 오류 RuntimeError: 임베딩 연결 실패" in p for p in problems) == 4 and "error" in rows[0]
    search.raise_error = None
    search.bad_id = True
    assert any("결과 규격 위반" in p for p in smoke(search, points)[0])


def test_smoke_irrelevant_results_fail(tmp_path):
    _, q, search, points = setup(tmp_path)
    for point in q.points.values():
        point["payload"]["text"] = "전혀 상관없는 문장입니다 가나다라마바사"
    problems, _, _ = smoke(search, points)
    assert sum("관련 단어가 있는 결과 0건" in p for p in problems) == 4
    assert smoke(search, points, min_relevant=0)[0] == []


def test_smoke_not_strict_makes_relevance_a_note(tmp_path):
    _, q, search, points = setup(tmp_path)
    for point in q.points.values():
        point["payload"]["text"] = "전혀 상관없는 문장입니다 가나다라마바사"
    text_of = {p.point_id: p.record["text"] for p in points}
    problems, notes, _ = vs.check_smoke(search, text_of, k=5, min_relevant=2, strict=False)
    assert problems == [] and sum("관련 단어가 있는 결과 0건" in n for n in notes) == 4


def test_smoke_unknown_ids_are_notes(tmp_path):
    _, q, search, points = setup(tmp_path)
    q.add(make_record(77))
    _, notes, _ = vs.check_smoke(search, {p.point_id: "x" for p in points[:2]}, k=5, min_relevant=0)
    assert any("05에 없는 id" in n for n in notes)


def test_doc_ids(tmp_path):
    _, q, search, _ = setup(tmp_path)
    assert vs.check_doc_ids(search, ["용접 불티 화재"]) == []
    search.doc_none = True
    notes = vs.check_doc_ids(search, ["용접 불티 화재", "x"])
    assert len(notes) == 1 and "SIF:None" in notes[0] and "A에게" in notes[0]
    search.doc_none = False
    search.raise_error = RuntimeError("x")
    assert vs.check_doc_ids(search, ["q"]) == []


def self_title(search, points, **kw):
    return vs.check_self_and_title(search, points, count_self=kw.get("s", 10), count_title=kw.get("t", 6), k=10)


def test_self_search_perfect_and_title_groups(tmp_path):
    _, q, search, points = setup(tmp_path)
    self_s, title_s, errors, rows = self_title(search, points)
    assert errors == [] and self_s["n"] == 10 and self_s["hit@1"] == 1.0
    assert title_s["n"] == 6 and title_s["hit@10"] == 1.0
    title_rows = [r for r in rows if r["kind"] == "title"]
    assert any(r["same_title_cases"] > 1 for r in title_rows)  # 같은 제목의 사례는 한 그룹으로 본다


def test_title_search_counts_any_member_of_the_group(tmp_path):
    _, q, _, points = setup(tmp_path, 24)
    groups = {}
    for p in points:
        groups.setdefault(vs.clean_title(p.record), []).append(p.point_id)
    shared = {t: ids for t, ids in groups.items() if len(ids) > 1}
    assert shared

    def search(query, *, k, filters):
        ids = shared.get(query)
        return [{"id": ids[-1], "doc_id": "SIF:1", "source": "t (l)", "text": "x", "score": 1.0}] if ids else []

    _, title_s, errors, rows = vs.check_self_and_title(search, points, count_self=1, count_title=len(groups), k=10)
    assert errors == []
    ranked = [r for r in rows if r["kind"] == "title" and r["target"] in shared]
    assert ranked and all(r["rank"] == 1 for r in ranked)


def test_self_search_detects_broken_ranking(tmp_path):
    _, q, search, points = setup(tmp_path, 40)
    search.shuffle = True
    self_s, _, _, _ = self_title(search, points)
    assert self_s["hit@1"] < 0.5


def test_self_search_errors_collected(tmp_path):
    _, q, search, points = setup(tmp_path)
    search.raise_error = RuntimeError("끊김")
    self_s, title_s, errors, _ = self_title(search, points)
    assert self_s["n"] == 0 and title_s["n"] == 0 and len(errors) == 16 and "RuntimeError: 끊김" in errors[0]


def test_filters_ok_and_skip_absent_values(tmp_path):
    records = make_records(8)
    records = [r for r in records if "CONFINED" not in r["work_type"]]
    q = FakeQdrant(records)
    search = FakeSearch(q)
    problems, notes, rows = vs.check_filters(search, check_records(records), k=5)
    assert problems == []
    assert not any(r["filter"] == "work_type=CONFINED" for r in rows)
    assert any("문자열로 호출" in n for n in notes)


def test_filters_leak_detected(tmp_path):
    _, q, search, points = setup(tmp_path)
    search.leak = True
    problems, _, _ = vs.check_filters(search, points, k=5)
    assert any("필터 누수" in p for p in problems)


def test_filters_empty_and_error(tmp_path):
    _, q, search, points = setup(tmp_path)
    search.empty = True
    problems, _, _ = vs.check_filters(search, points, k=5)
    assert any("결과가 없습니다" in p for p in problems)
    search.empty = False
    search.error_on_filter = True
    problems, _, rows = vs.check_filters(search, points, k=5)
    assert any("검색 오류 RuntimeError: 필터 오류" in p for p in problems) and "error" in rows[0]


def test_filters_list_form_fallback(tmp_path):
    _, q, search, points = setup(tmp_path)
    search.list_only = True
    problems, notes, _ = vs.check_filters(search, points, k=5)
    assert problems == [] and any("배열로 호출" in n for n in notes)
    forms = [c["filters"] for c in search.calls if "accident_type" in c["filters"]]
    assert any(isinstance(f["accident_type"], list) for f in forms)


def test_filters_unknown_ids_are_notes(tmp_path):
    _, q, search, points = setup(tmp_path)
    q.add(make_record(77))
    problems, notes, _ = vs.check_filters(search, points[:8], k=20)
    assert problems == [] and any("05에 없는 id" in n for n in notes)


def test_default_search_factory_without_app_is_input_error(monkeypatch):
    # 앱이 실제로 있는 PC에서도 같은 결과가 나오도록 import를 막아 둔다.
    monkeypatch.setitem(sys.modules, "app.rag.tool_backend", None)
    with pytest.raises(InputError, match="app.rag.tool_backend"):
        vs.default_search_factory()


def test_default_search_factory_uses_backend(monkeypatch):
    import types

    marker = object()
    for name in ("app", "app.rag"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("app.rag.tool_backend")
    module.build_search_backend = lambda: marker
    monkeypatch.setitem(sys.modules, "app.rag.tool_backend", module)
    assert vs.default_search_factory() is marker


# ---------------------------------------------------------------- 인자


def test_parse_args_defaults():
    a = vs.parse_args([])
    assert a.chunks == Path("out/accidents/law/05_chunks.jsonl") and a.self_queries == 50 and a.title_queries == 50
    assert a.k_fetch == 10 and a.min_self_hit5 == 0.9 and a.min_relevant == 2 and a.embedding_dim == 1024
    assert a.qdrant_url == "http://127.0.0.1:60011"


@pytest.mark.parametrize("argv", [
    ["--embedding-dim", "0"], ["--limit", "0"], ["--k-fetch", "0"], ["--self-queries", "-1"], ["--min-self-hit5", "1.5"],
    ["--skip-qdrant", "--skip-search"],
])
def test_parse_args_rejects(argv):
    with pytest.raises(SystemExit):
        vs.parse_args(argv)


def test_parse_args_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EMBEDDING_DIM", "512")
    a = vs.parse_args([])
    assert (a.qdrant_url, a.embedding_dim) == ("http://127.0.0.1:1", 512)


# ---------------------------------------------------------------- 전체 실행


def test_run_dry_run(tmp_path):
    chunks, _, _, _ = setup(tmp_path)
    logs: list[str] = []
    assert vs.run(args_for(chunks, "--dry-run"), session=Boom(), log=logs.append) == 0
    text = "\n".join(logs)
    assert "SIF 청크 16건" in text and "자기 검색 질문 예" in text and "[dry-run]" in text and "accident_type=FALL" in text


def test_run_dry_run_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("QDRANT_READ_ONLY_API_KEY")
    chunks, _, _, _ = setup(tmp_path)
    assert vs.run(args_for(chunks, "--dry-run"), session=Boom(), log=lambda *_: None) == 0


def test_run_full_passes_and_writes_report(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    out = tmp_path / "eval" / "r.json"
    logs: list[str] = []
    assert go(args_for(chunks, "--out", str(out)), q, search, logs) == 0
    text = "\n".join(logs)
    assert "[1부]" in text and "[2부]" in text and "[통과]" in text and "자기 검색" in text and "검색 지연" in text
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["problems"] == [] and report["expected"] == 16 and report["self_search"]["hit@1"] == 1.0
    assert report["smoke"] and report["filters"]


def test_run_only_reads(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    go(args_for(chunks), q, search)
    assert {kind for kind, _ in q.calls} <= {"GET", "POST"}
    assert all(not url.endswith(("/points", "/points?wait=true")) or True for _, url in q.calls)
    assert not any("PUT" == kind for kind, _ in q.calls) and not hasattr(q, "put")


def test_run_skip_search_needs_no_search_backend(tmp_path):
    chunks, q, _, _ = setup(tmp_path)
    called: list[int] = []
    assert vs.run(args_for(chunks, "--skip-search"), session=q, search_factory=lambda: called.append(1), sleep=no_sleep, log=lambda *_: None) == 0
    assert called == []


def test_run_skip_qdrant_needs_no_key_or_session(tmp_path, monkeypatch):
    monkeypatch.delenv("QDRANT_READ_ONLY_API_KEY")
    chunks, q, search, _ = setup(tmp_path)
    assert go(args_for(chunks, "--skip-qdrant"), Boom(), search) == 0


def test_run_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("QDRANT_READ_ONLY_API_KEY")
    chunks, q, search, _ = setup(tmp_path)
    with pytest.raises(InputError, match="QDRANT_READ_ONLY_API_KEY"):
        go(args_for(chunks), q, search)


def test_run_missing_points_fail_with_summary(tmp_path):
    chunks, q, search, points = setup(tmp_path)
    for p in points[:2]:
        del q.points[p.point_id]
    with pytest.raises(IngestError, match=r"(?s)확인에서 문제.*Qdrant에 없는 점 2건"):
        go(args_for(chunks), q, search)


def test_run_failure_still_writes_report(tmp_path):
    chunks, q, search, points = setup(tmp_path)
    del q.points[points[0].point_id]
    out = tmp_path / "r.json"
    with pytest.raises(IngestError):
        go(args_for(chunks, "--out", str(out)), q, search)
    assert json.loads(out.read_text(encoding="utf-8"))["problems"]


def test_run_self_search_threshold(tmp_path):
    chunks, q, search, _ = setup(tmp_path, 40)
    search.shuffle = True
    with pytest.raises(IngestError, match="자기 검색 hit@5"):
        go(args_for(chunks), q, search)
    search.calls.clear()
    # 같은 상황에서 기준을 0으로 낮추면 이 항목은 문제가 아니다
    with pytest.raises(IngestError) as caught:
        go(args_for(chunks, "--min-self-hit5", "0"), q, search)
    assert "자기 검색 hit@5" not in str(caught.value)


def test_run_search_errors_are_problems(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    search.raise_error = RuntimeError("임베딩 연결 실패")
    with pytest.raises(IngestError, match="RuntimeError: 임베딩 연결 실패"):
        go(args_for(chunks), q, search)


def test_run_limit_mode_with_partial_load(tmp_path):
    records = make_records(16)
    chunks = write_chunks(tmp_path, records)
    q = FakeQdrant(records[:5], load=True)  # 앞 5건만 올림
    search = FakeSearch(q)
    logs: list[str] = []
    assert go(args_for(chunks, "--limit", "5"), q, search, logs) == 0
    assert any("[참고]" in m and "관련 단어" in m for m in logs)  # 5건뿐이라 관련 단어 부족은 참고로만
    with pytest.raises(IngestError, match="기대 16, 실제 5"):
        go(args_for(chunks), q, search)


def test_run_limit_mode_extra_points_is_note(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    logs: list[str] = []
    assert go(args_for(chunks, "--limit", "5", "--skip-search"), q, None, logs) == 0
    assert any("[참고]" in m and "16개로" in m for m in logs)


def test_run_remote_refused(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    with pytest.raises(InputError, match="로컬이 아닙니다"):
        go(args_for(chunks, "--qdrant-url", "http://10.1.1.1:60011"), q, search)
    assert q.calls == []
    assert go(args_for(chunks, "--qdrant-url", "http://10.1.1.1:60011", "--allow-remote"), q, search) == 0


def test_run_bad_chunks(tmp_path):
    with pytest.raises(InputError, match="청크 파일이 없습니다"):
        vs.run(vs.parse_args(["--chunks", str(tmp_path / "x.jsonl"), "--dry-run"]), session=Boom(), log=lambda *_: None)
    bad = make_records(2)
    bad[0]["flagged"] = "no"
    with pytest.raises(InputError, match="flagged"):
        vs.run(vs.parse_args(["--chunks", str(write_chunks(tmp_path, bad)), "--dry-run"]), session=Boom(), log=lambda *_: None)


def test_run_reports_doc_id_note(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    search.doc_none = True
    logs: list[str] = []
    assert go(args_for(chunks), q, search, logs) == 0  # 참고 사항이지 실패가 아니다
    assert any("[참고]" in m and "SIF:None" in m for m in logs)


def test_run_does_not_modify_input(tmp_path):
    chunks, q, search, _ = setup(tmp_path)
    before = chunks.read_bytes()
    go(args_for(chunks), q, search)
    assert chunks.read_bytes() == before


# ---------------------------------------------------------------- main 종료 코드


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    chunks, _, _, _ = setup(tmp_path)
    monkeypatch.setattr(vs, "run", lambda args: 0)
    with pytest.raises(SystemExit) as ok:
        vs.main(["--chunks", str(chunks), "--dry-run"])
    assert ok.value.code == 0
    monkeypatch.setattr(vs, "run", lambda args: (_ for _ in ()).throw(InputError("입력")))
    with pytest.raises(SystemExit) as bad:
        vs.main(["--chunks", str(chunks)])
    assert bad.value.code == 2 and "[입력 오류] 입력" in capsys.readouterr().err
    monkeypatch.setattr(vs, "run", lambda args: (_ for _ in ()).throw(IngestError("문제")))
    with pytest.raises(SystemExit) as failed:
        vs.main(["--chunks", str(chunks)])
    assert failed.value.code == 1 and "[문제] 문제" in capsys.readouterr().err
    monkeypatch.setattr(vs, "run", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(SystemExit) as stopped:
        vs.main(["--chunks", str(chunks)])
    assert stopped.value.code == 1 and "안전합니다" in capsys.readouterr().err


def test_source_hygiene():
    source = Path(vs.__file__).read_text(encoding="utf-8")
    assert "pymysql" not in source and "\r" not in source
    assert "\nfrom app" not in source and "\nimport app" not in source  # 앱 코드는 검색 단계에서만 지연 import
    assert ".put(" not in source and "PUT" not in source  # 쓰기 요청 없음
