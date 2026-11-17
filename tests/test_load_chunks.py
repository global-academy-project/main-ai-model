"""scripts/load_chunks.py 시험. 서버는 test_ingest_via_api.py 의 스텁(가짜 kb-ingest)을 재사용한다."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # load_chunks 가 ingest_via_api 를 import 한다
sys.path.insert(0, str(ROOT / "tests"))

from test_ingest_via_api import Stub, TOKEN  # noqa: E402  (스텁 재사용)

spec = importlib.util.spec_from_file_location("load_chunks", ROOT / "scripts" / "load_chunks.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

from demo_data import LAW_REQ, SAMPLE, SIF_REQ  # noqa: E402  (samples 폴더 없이 시험 안에 담은 데모 데이터)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KB_INGEST_API_TOKEN", raising=False)


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def jl(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                 encoding="utf-8")
    return p


def sample_rows():
    return [json.loads(x) for x in SAMPLE.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(args):
    return lc.main([str(a) for a in args])


def build(paths, defaults=None, only=None):
    return lc.build_requests(paths, defaults or {}, only)


# ---------------------------------------------------------------- 변환

def test_sample_builds_expected_requests():
    items, errors = build([SAMPLE])
    assert errors == []
    names = [n for n, _ in items]
    assert names == ["chunks_demo.jsonl:SIF-DEMO-001", "chunks_demo.jsonl:LAW-DEMO-001"]
    # SIF 한 줄은 API 문서의 요청 본문과 정확히 같아야 한다
    assert items[0][1] == SIF_REQ
    law = items[1][1]
    assert [c["chunk_index"] for c in law["chunks"]] == [1, 2]
    assert law["chunks"][0] == LAW_REQ["chunks"][0]
    assert law["source"] == LAW_REQ["source"]  # 2번째 줄은 원본 정보를 안 적어도 이어받는다


def test_groups_by_source_and_keeps_first_seen_order(tmp_path):
    a, b = sample_rows()[0], sample_rows()[1]
    a2 = dict(a, chunk_index=1, text="두 번째")
    p = jl(tmp_path, "x.jsonl", [a, b, a2])
    items, errors = build([p])
    assert errors == []
    assert [n.split(":")[1] for n, _ in items] == ["SIF-DEMO-001", "LAW-DEMO-001"]
    assert [c["chunk_index"] for c in items[0][1]["chunks"]] == [0, 1]


def test_same_external_id_in_two_files_is_one_source(tmp_path):
    rows = sample_rows()
    a = jl(tmp_path, "a.jsonl", [rows[0]])
    b = jl(tmp_path, "b.jsonl", [dict(rows[0], chunk_index=1, text="다음 파일")])
    items, errors = build([a, b])
    assert errors == [] and len(items) == 1 and len(items[0][1]["chunks"]) == 2


def test_cli_defaults_fill_only_missing_keys(tmp_path):
    row = {k: v for k, v in sample_rows()[0].items() if k not in ("source_type", "publisher", "license")}
    row["license"] = "줄 값이 우선"
    items, errors = build([jl(tmp_path, "d.jsonl", [row])],
                          {"source_type": "SIF", "publisher": "기본 발행처", "license": "기본 라이선스"})
    assert errors == []
    src = items[0][1]["source"]
    assert src["source_type"] == "SIF" and src["publisher"] == "기본 발행처" and src["license"] == "줄 값이 우선"


def test_only_filters_sources_and_reports_missing(tmp_path):
    items, errors = build([SAMPLE], only={"LAW-DEMO-001"})
    assert [n.split(":")[1] for n, _ in items] == ["LAW-DEMO-001"] and errors == []
    _, errors = build([SAMPLE], only={"NOPE"})
    assert any("NOPE" in e for e in errors)


# ---------------------------------------------------------------- 입력 오류는 줄 번호와 함께 모두 모은다

def test_errors_are_collected_with_line_numbers(tmp_path):
    good = sample_rows()[0]
    p = jl(tmp_path, "bad.jsonl", [
        good,
        "{not json",
        json.dumps([1, 2]),
        dict(good, embed_text="x"),                       # 알 수 없는 키(이름이 다른 필드)
        {k: v for k, v in good.items() if k != "external_id"},
        dict(good, chunk_index=5, title="다른 제목"),        # 같은 원본인데 title 이 다름
    ])
    _, errors = build([p])
    text = "\n".join(errors)
    assert "bad.jsonl:2" in text and "JSON 이 아니다" in text
    assert "bad.jsonl:3" in text and "객체가 아니다" in text
    assert "bad.jsonl:4" in text and "embedding_text 인가?" in text
    assert "bad.jsonl:5" in text and "external_id" in text
    assert "bad.jsonl:6" in text and "title" in text


def test_missing_file_and_missing_source_type(tmp_path):
    _, errors = build([tmp_path / "none.jsonl"])
    assert any("파일이 없다" in e for e in errors)
    row = {k: v for k, v in sample_rows()[0].items() if k != "source_type"}
    _, errors = build([jl(tmp_path, "n.jsonl", [row])])
    assert any("source_type 가 없다" in e for e in errors)


def test_blank_lines_and_bom_are_ok(tmp_path):
    p = tmp_path / "bom.jsonl"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(sample_rows()[0], ensure_ascii=False).encode("utf-8") + b"\n\n\n")
    items, errors = build([p])
    assert errors == [] and len(items) == 1


# ---------------------------------------------------------------- 실행 흐름

def test_default_validates_only_without_network(capsys):
    assert run([SAMPLE, "--base-url", "http://127.0.0.1:9"]) == 0
    out = capsys.readouterr().out
    assert "원본 2개, 청크 3개" in out and "전송 안 함" in out


def test_input_errors_block_everything(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    p = jl(tmp_path, "bad.jsonl", [sample_rows()[0], "{oops"])
    assert run([p, "--send", "--base-url", stub.url]) == 2
    assert stub.calls == []
    assert "bad.jsonl:2" in capsys.readouterr().out


def test_semantic_errors_from_core_validation_block_everything(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    rows = sample_rows()
    rows[0]["payload"]["accident_type"] = ["감전"]  # 한글 라벨은 코어 검증이 막는다
    p = jl(tmp_path, "sem.jsonl", rows)
    assert run([p, "--send", "--base-url", stub.url]) == 2
    assert stub.calls == []


def test_send_posts_one_request_per_source(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    report = tmp_path / "r.jsonl"
    assert run([SAMPLE, "--send", "--base-url", stub.url, "--report", report]) == 0
    posts = stub.posts()
    assert [c[1] for c in posts] == ["/ingest/accidents", "/ingest/guides"]
    bodies = [json.loads(c[3].decode("utf-8")) for c in posts]
    assert bodies[0] == SIF_REQ and len(bodies[1]["chunks"]) == 2
    rows = [json.loads(x) for x in report.read_text(encoding="utf-8").splitlines()]
    assert [r["sent_chunks"] for r in rows] == [1, 2] and all(r["ok"] for r in rows)
    assert TOKEN not in report.read_text(encoding="utf-8") + capsys.readouterr().out


def test_only_sends_selected_source(stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    assert run([SAMPLE, "--only", "SIF-DEMO-001", "--send", "--base-url", stub.url]) == 0
    assert [c[1] for c in stub.posts()] == ["/ingest/accidents"]


def test_failure_exit_code_and_stop(stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    stub.mode = "502"
    assert run([SAMPLE, "--send", "--base-url", stub.url]) == 1
    assert len(stub.posts()) == 1
    stub.calls.clear()
    assert run([SAMPLE, "--send", "--keep-going", "--base-url", stub.url]) == 1
    assert len(stub.posts()) == 2


def test_remote_guard_and_token_still_apply(tmp_path, stub, monkeypatch):
    assert run([SAMPLE, "--send", "--base-url", stub.url]) == 2  # 토큰 없음
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    assert run([SAMPLE, "--send", "--base-url", stub.alt_url]) == 2  # 로컬 아님
    assert stub.calls == []
