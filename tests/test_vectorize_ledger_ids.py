"""vectorize_accidents 벡터 파일 검사: 장부 id(kb_source_id·kb_chunk_id) 규칙.

서버·Qdrant 없이 임시 벡터 파일만으로 돈다. 기존 tests/test_vectorize_accidents.py 와 겹치지 않게 별도 파일로 둔다.
"""

from __future__ import annotations

import json

import pytest

from scripts import vectorize_accidents as va

DIM = 3


def point(n: int) -> va.ExpectedPoint:
    return va.ExpectedPoint({}, va.point_id_for(f"SIF-{n:04d}", 0), f"SIF-{n:04d}/0")


def write(tmp_path, points, payloads):
    path = tmp_path / "v.jsonl"
    lines = []
    for p, payload in zip(points, payloads):
        item = {
            "id": p.point_id,
            "vector": {"dense": [0.1, 0.2, 0.3], "sparse": {"indices": [1, 5], "values": [0.5, 0.25]}},
            "payload": payload,
        }
        lines.append(json.dumps(item))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def check(tmp_path, payloads, **kw):
    points = [point(i) for i in range(len(payloads))]
    return va.check_vector_file(write(tmp_path, points, payloads), points, DIM, **kw)


def ids(source, chunk):
    return {"kb_source_id": source, "kb_chunk_id": chunk}


def test_old_server_null_ids_pass(tmp_path):
    seen, problems = check(tmp_path, [ids(None, None), {}])      # 키가 없는 것도 null 과 같다
    assert (seen, problems) == (2, [])


def test_new_server_integer_ids_pass(tmp_path):
    # PC 에서 실제로 나온 값 (11/12): kb_source_id 11,12,13 / kb_chunk_id 3495,3496,3497
    seen, problems = check(tmp_path, [ids(11, 3495), ids(12, 3496), ids(13, 3497)])
    assert (seen, problems) == (3, [])


@pytest.mark.parametrize("bad", [0, -1, "11", 1.5, True, [11]])
def test_bad_values_are_reported(tmp_path, bad):
    _, problems = check(tmp_path, [ids(bad, 3495)])
    assert len(problems) == 1 and "양의 정수나 null이 아님" in problems[0] and "kb_source_id" in problems[0]


def test_only_one_of_the_two_is_reported(tmp_path):
    _, problems = check(tmp_path, [ids(11, None)])
    assert len(problems) == 1 and "하나만 있음" in problems[0]


def test_duplicate_chunk_id_is_reported(tmp_path):
    _, problems = check(tmp_path, [ids(11, 3495), ids(12, 3495)])
    assert len(problems) == 1 and "kb_chunk_id 3495가 겹침" in problems[0]


def test_same_source_id_with_different_chunks_is_fine(tmp_path):
    # 소스 하나에 청크 여러 개인 자료(법령 등)가 있어도 소스 id 중복은 문제 아니다
    _, problems = check(tmp_path, [ids(11, 3495), ids(11, 3496)])
    assert problems == []


def test_require_ledger_ids_rejects_null(tmp_path):
    _, problems = check(tmp_path, [ids(11, 3495), ids(None, None)], require_ledger_ids=True)
    assert len(problems) == 1 and "--require-ledger-ids" in problems[0]


def test_require_ledger_ids_accepts_filled(tmp_path):
    assert check(tmp_path, [ids(11, 3495), ids(12, 3496)], require_ledger_ids=True)[1] == []


def test_other_checks_still_work(tmp_path):
    points = [point(0), point(1)]
    path = write(tmp_path, points[:1], [ids(11, 3495)])          # 두 번째 점이 파일에 없다
    seen, problems = va.check_vector_file(path, points, DIM)
    assert seen == 1 and len(problems) == 1 and "벡터 파일에 없는 청크" in problems[0]


def test_ledger_id_counts(tmp_path):
    points = [point(i) for i in range(3)]
    path = write(tmp_path, points, [ids(11, 3495), ids(None, None), {}])
    assert va.ledger_id_counts(path) == (1, 2)


def test_option_default_is_off():
    args = va.parse_args(["--dry-run"])
    assert args.require_ledger_ids is False
    assert va.parse_args(["--require-ledger-ids"]).require_ledger_ids is True


# ── run() 전체 흐름 (--check-only): PC 에서 실제로 났던 상황을 재현 ───────────────────────

def _record(n: int) -> dict:
    ext = f"SIF-{n:04d}"
    return {
        "natural_key": {"dataset": "sif", "record": ext, "section": "body", "part": 0},
        "external_id": ext, "chunk_index": 0, "source_type": "SIF", "collection": "kb_accidents",
        "text": "본문", "embed_text": "임베딩 문장", "site_id": 0, "flagged": False,
        "title": "제목", "locator": "위치", "year": 2024, "severity": "FATAL",
        "accident_type": [], "hazard": [], "work_type": [], "cas": [], "ghs": [],
    }


def _run_check_only(tmp_path, monkeypatch, ledger_for):
    records = [_record(i) for i in range(3)]
    monkeypatch.setattr(va, "load_records", lambda path: records)
    chunks = tmp_path / "05_chunks.jsonl"
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    lines = []
    for i, record in enumerate(records):
        payload = dict(record)
        payload.update(ledger_for(i))
        lines.append(json.dumps({
            "id": va.point_id_for(record["external_id"], 0),
            "vector": {"dense": [0.1] * 4, "sparse": {"indices": [1], "values": [0.5]}},
            "payload": payload,
        }))
    (vector_dir / "ds.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = va.parse_args(["--chunks", str(chunks), "--dataset-id", "ds", "--vector-dir", str(vector_dir),
                          "--embedding-dim", "4", "--check-only", "--no-validation"])
    out: list[str] = []
    return va.run(args, log=out.append), "\n".join(out)


def test_run_passes_with_new_server_ids(tmp_path, monkeypatch):
    code, text = _run_check_only(tmp_path, monkeypatch, lambda i: ids(11 + i, 3495 + i))
    assert code == 0
    assert "장부 id(kb_source_id·kb_chunk_id): 있음 3점 / 없음 0점" in text
    assert "doc_id가 비게" not in text


def test_run_passes_with_old_server_but_warns(tmp_path, monkeypatch):
    code, text = _run_check_only(tmp_path, monkeypatch, lambda i: ids(None, None))
    assert code == 0
    assert "있음 0점 / 없음 3점" in text and "doc_id가 비게" in text


def test_run_fails_when_ids_required_but_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "load_records", lambda path: [_record(0)])
    chunks = tmp_path / "05_chunks.jsonl"
    vector_dir = tmp_path / "vectors"
    vector_dir.mkdir()
    payload = dict(_record(0), kb_source_id=None, kb_chunk_id=None)
    (vector_dir / "ds.jsonl").write_text(json.dumps({
        "id": va.point_id_for("SIF-0000", 0),
        "vector": {"dense": [0.1] * 4, "sparse": {"indices": [1], "values": [0.5]}}, "payload": payload,
    }) + "\n", encoding="utf-8")
    args = va.parse_args(["--chunks", str(chunks), "--dataset-id", "ds", "--vector-dir", str(vector_dir),
                          "--embedding-dim", "4", "--check-only", "--no-validation", "--require-ledger-ids"])
    with pytest.raises(va.IngestError, match="--require-ledger-ids"):
        va.run(args, log=lambda s: None)
