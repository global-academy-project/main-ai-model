"""scripts/ingest_via_api.py 시험. 서버는 스텁(가짜 kb-ingest)이다. 실제 kb-ingest·Qdrant·MySQL은 쓰지 않는다."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest_via_api", ROOT / "scripts" / "ingest_via_api.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sys.path.insert(0, str(ROOT / "tests"))
from demo_data import LAW_REQ as LAW, SIF_REQ as SIF  # noqa: E402  (samples 폴더 없이 시험 안에 담은 데모 데이터)
TOKEN = "test-token-value"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KB_INGEST_API_TOKEN", raising=False)


class Stub:
    """가짜 kb-ingest. behavior 로 응답을 바꾼다."""

    def __init__(self):
        self.calls = []  # (method, path, headers, body)
        self.scheme = {"type": "http", "scheme": "bearer"}
        self.paths = ["/ingest/accidents", "/ingest/guides"]
        self.ready = 200
        self.mode = "ok"  # ok | 422 | 502 | bad_indexed | bad_point
        self.token = TOKEN
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.calls.append(("GET", self.path, dict(self.headers), b""))
                if self.path == "/ready":
                    return self._send(outer.ready, {"status": "ok"})
                if self.path == "/openapi.json":
                    return self._send(200, {
                        "paths": {p: {} for p in outer.paths},
                        "components": {"securitySchemes": {"auth": outer.scheme}},
                    })
                self._send(404, {"detail": "nope"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n)
                outer.calls.append(("POST", self.path, dict(self.headers), body))
                if outer.scheme["type"] == "http":
                    ok = self.headers.get("Authorization") == f"Bearer {outer.token}"
                else:
                    ok = self.headers.get(outer.scheme["name"]) == outer.token
                if not ok:
                    return self._send(401, {"detail": "Unauthorized"})
                if outer.mode == "422":
                    return self._send(422, {"detail": "bad chunk"})
                if outer.mode == "502":
                    return self._send(502, {"detail": "embedding down"})
                req = json.loads(body.decode("utf-8"))
                st, ext = req["source"]["source_type"], req["source"]["external_id"]
                chunks = []
                for i, c in enumerate(req["chunks"]):
                    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{st}:{ext}:{c['chunk_index']}"))
                    if outer.mode == "bad_point":
                        pid = str(uuid.uuid4())
                    chunks.append({"chunk_index": c["chunk_index"], "chunk_id": 100 + i, "point_id": pid})
                indexed = len(chunks) - (1 if outer.mode == "bad_indexed" else 0)
                self._send(200, {"indexed": indexed, "source_id": 7, "chunks": chunks})

        self.server = HTTPServer(("0.0.0.0", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.alt_url = f"http://127.0.0.2:{self.server.server_port}"  # 로컬이 아닌 주소로 취급되지만 스텁에 닿는다
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def run(args):
    return mod.main([str(a) for a in args])


# ---------------------------------------------------------------- 검증

def errs(req, **kw):
    return mod.validate_request(req, **kw)[0]


def test_demo_samples_are_valid():
    assert errs(SIF) == []
    assert errs(LAW) == []


def test_endpoint_follows_source_type():
    assert mod.validate_request(SIF)[2]["endpoint"] == "/ingest/accidents"
    assert mod.validate_request(LAW)[2]["endpoint"] == "/ingest/guides"
    r = copy.deepcopy(SIF)
    r["source"]["source_type"] = "ACCIDENT_REPORT"
    assert mod.validate_request(r)[2]["endpoint"] == "/ingest/accidents"


@pytest.mark.parametrize("stype", ["NEAR_MISS", "AUDIT", "", None])
def test_rejects_unknown_source_type(stype):
    r = copy.deepcopy(SIF)
    r["source"]["source_type"] = stype
    assert any("source_type" in e for e in errs(r))


def test_requires_url_or_file_path():
    r = copy.deepcopy(SIF)
    del r["source"]["url"]
    assert any("url 또는" in e for e in errs(r))
    r["source"]["file_path"] = "raw/x.pdf"
    assert errs(r) == []


def test_duplicate_chunk_index_and_blank_text():
    r = copy.deepcopy(SIF)
    r["chunks"].append(copy.deepcopy(r["chunks"][0]))
    assert any("중복" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["text"] = "   "
    assert any("text 가 비어" in e for e in errs(r))


def test_empty_and_too_many_chunks():
    r = copy.deepcopy(SIF)
    r["chunks"] = []
    assert any("비어" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    base = r["chunks"][0]
    r["chunks"] = [dict(base, chunk_index=i) for i in range(mod.MAX_CHUNKS + 1)]
    assert any("최대" in e for e in errs(r))


def test_many_chunks_only_warns():
    r = copy.deepcopy(SIF)
    base = r["chunks"][0]
    r["chunks"] = [dict(base, chunk_index=i) for i in range(9)]
    e, w, _ = mod.validate_request(r)
    assert e == [] and any("502" in x for x in w)


def test_tag_codes_and_shapes():
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["accident_type"] = ["감전"]  # 한글 라벨 금지
    assert any("스냅샷" in e for e in errs(r))
    assert errs(r, allow_unknown_tags=True) == []
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["hazard"] = "ELECTRICAL"  # 배열이어야 함
    assert any("문자열 배열" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["severity"] = ["FATAL"]
    assert any("severity" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["cas"] = ["64-17-5", "abc"]
    assert any("CAS" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["year"] = "2018"
    assert any("year" in e for e in errs(r))


def test_sif_requires_its_fields():
    r = copy.deepcopy(SIF)
    del r["chunks"][0]["payload"]["tier"]
    assert any("tier" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["situation"] = None  # 키가 있고 값이 null 인 것은 정상
    assert errs(r) == []


def test_law_rules():
    r = copy.deepcopy(LAW)
    r["chunks"][0]["payload"]["natural_key"]["law_key"] = "다른법"
    assert any("law_key" in e for e in errs(r))
    r = copy.deepcopy(LAW)
    two = dict(copy.deepcopy(r["chunks"][0]), chunk_index=2)
    two["payload"]["law_meta"]["source_sha256"] = "d" * 64
    r["chunks"].append(two)
    assert any("source_sha256 값이 서로" in e for e in errs(r))
    r = copy.deepcopy(LAW)
    del r["chunks"][0]["payload"]["article_sha256"]
    assert any("article_sha256" in e for e in errs(r))


def test_server_fields_warn_only():
    r = copy.deepcopy(SIF)
    r["chunks"][0]["payload"]["kb_source_id"] = 5
    e, w, _ = mod.validate_request(r)
    assert e == [] and any("덮어쓴다" in x for x in w)


def test_length_limits():
    r = copy.deepcopy(SIF)
    r["source"]["external_id"] = "x" * 101
    assert any("external_id" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["source"]["content_sha256"] = "A" * 64
    assert any("content_sha256" in e for e in errs(r))
    r = copy.deepcopy(SIF)
    r["chunks"][0]["locator"] = "가" * 101
    assert any("locator" in e for e in errs(r))


# ---------------------------------------------------------------- 실행 흐름

def test_default_is_validate_only_and_touches_no_network(tmp_path, capsys):
    p = write(tmp_path, "a.json", SIF)
    # 닫힌 포트: 네트워크를 건드리면 실패한다
    assert run([p, "--base-url", "http://127.0.0.1:9"]) == 0
    assert "전송 안 함" in capsys.readouterr().out


def test_invalid_file_blocks_everything(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    good = write(tmp_path, "good.json", SIF)
    bad_req = copy.deepcopy(LAW)
    bad_req["chunks"][0]["text"] = ""
    bad = write(tmp_path, "bad.json", bad_req)
    assert run([good, bad, "--send", "--base-url", stub.url]) == 2
    assert stub.calls == []  # 좋은 파일도 보내지 않는다


def test_send_success_routes_by_type_and_reports(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    a = write(tmp_path, "a.json", SIF)
    b = write(tmp_path, "b.json", LAW)
    report = tmp_path / "report.jsonl"
    assert run([a, b, "--send", "--base-url", stub.url, "--report", report]) == 0
    assert [c[1] for c in stub.posts()] == ["/ingest/accidents", "/ingest/guides"]
    rows = [json.loads(x) for x in report.read_text(encoding="utf-8").splitlines()]
    assert [r["ok"] for r in rows] == [True, True]
    assert rows[0]["source_id"] == 7 and rows[0]["indexed"] == 1
    assert TOKEN not in report.read_text(encoding="utf-8")
    assert TOKEN not in capsys.readouterr().out
    # 본문은 UTF-8 한글 그대로
    assert "거푸집" in stub.posts()[0][3].decode("utf-8")


def test_apikey_header_scheme_is_detected(tmp_path, stub, monkeypatch):
    stub.scheme = {"type": "apiKey", "in": "header", "name": "X-KB-Token"}
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    assert run([write(tmp_path, "a.json", SIF), "--send", "--base-url", stub.url]) == 0
    sent = {k.lower(): v for k, v in stub.posts()[0][2].items()}  # urllib 은 헤더 이름 대소문자를 바꾼다
    assert sent.get("x-kb-token") == TOKEN
    assert "authorization" not in sent


def test_wrong_token_gives_401_hint(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", "wrong")
    assert run([write(tmp_path, "a.json", SIF), "--send", "--base-url", stub.url]) == 1
    out = capsys.readouterr().out
    assert "HTTP 401" in out and "KB_INGEST_API_TOKEN" in out
    assert "wrong" not in out


def test_missing_token_is_config_error(tmp_path, stub):
    assert run([write(tmp_path, "a.json", SIF), "--send", "--base-url", stub.url]) == 2
    assert stub.calls == []


def test_remote_target_needs_flag(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    p = write(tmp_path, "a.json", SIF)
    assert run([p, "--send", "--base-url", stub.alt_url]) == 2
    assert stub.calls == []  # 연결 시도조차 하지 않는다
    assert run([p, "--send", "--allow-remote", "--base-url", stub.alt_url]) == 0
    assert len(stub.posts()) == 1


def test_not_ready_and_missing_endpoint(tmp_path, stub, monkeypatch):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    p = write(tmp_path, "a.json", SIF)
    stub.ready = 503
    assert run([p, "--send", "--base-url", stub.url]) == 2
    stub.ready = 200
    stub.paths = ["/ingest/guides"]
    assert run([p, "--send", "--base-url", stub.url]) == 2
    assert stub.posts() == []


def test_server_errors_exit_1_and_stop(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    a, b = write(tmp_path, "a.json", SIF), write(tmp_path, "b.json", LAW)
    stub.mode = "422"
    assert run([a, b, "--send", "--base-url", stub.url]) == 1
    assert len(stub.posts()) == 1  # 기본은 첫 실패에서 멈춘다
    assert "HTTP 422" in capsys.readouterr().out
    stub.calls.clear()
    stub.mode = "502"
    assert run([a, b, "--send", "--keep-going", "--base-url", stub.url]) == 1
    assert len(stub.posts()) == 2


def test_response_mismatch_is_failure(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    p = write(tmp_path, "a.json", SIF)
    stub.mode = "bad_indexed"
    assert run([p, "--send", "--base-url", stub.url]) == 1
    assert "indexed=0" in capsys.readouterr().out


def test_unexpected_point_id_only_warns(tmp_path, stub, monkeypatch, capsys):
    monkeypatch.setenv("KB_INGEST_API_TOKEN", TOKEN)
    p = write(tmp_path, "a.json", SIF)
    stub.mode = "bad_point"
    assert run([p, "--send", "--base-url", stub.url]) == 0
    assert "예상 UUID v5" in capsys.readouterr().out


def test_unreadable_inputs(tmp_path):
    assert run([tmp_path / "none.json"]) == 2
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert run([broken]) == 2


def test_bom_file_is_read(tmp_path):
    p = tmp_path / "bom.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(SIF, ensure_ascii=False).encode("utf-8"))
    assert run([p]) == 0
