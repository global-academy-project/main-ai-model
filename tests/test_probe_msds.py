"""scripts/probe_msds.py 시험. 공단 API 대신 127.0.0.1의 가짜 서버를 쓴다 (실제 API·키·외부망 없음)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from app.agent.text_rules import MATERIALS
from scripts import probe_msds

SECRET = "TEST-SECRET-KEY-123"

# 목록 API가 돌려줄 후보 (이름 → [(CAS, chemId)])
CHEMICALS = {
    "톨루엔": [("108-88-3", "T1")],
    "아세톤": [("67-64-1", "A1")],
    "크실렌": [("1330-20-7", "X1")],
    "모호": [("111-11-1", "M1"), ("222-22-2", "M2")],  # 같은 이름에 CAS 둘
    "벤젠류": [("71-43-2", "B1")],  # '벤젠'과 이름 불일치
    "이소프로필 알코올": [("67-63-0", "I1")],  # 공백 때문에 글자는 다르지만 이름 비교(공백 무시)에는 맞는다
    "Toluene": [("108-88-3", "T2")],
}
DETAIL = {
    "getChemDetail021": [("유해성·위험성 분류", "인화성 액체 : 구분2"), ("유해·위험 문구", "고인화성 액체 및 증기")],
    "getChemDetail041": [("흡입했을 때", "신선한 공기로 옮긴다.")],
    "getChemDetail151": [("산업안전보건법에 의한 규제", "관리대상유해물질"), ("위험물안전관리법에 의한 규제", "제4류 제1석유류")],
}


def _xml(items: list[dict[str, str]], code: str = "00") -> bytes:
    body = "".join("<item>" + "".join(f"<{k}>{v}</{k}>" for k, v in item.items()) + "</item>" for item in items)
    return (
        f"<response><header><resultCode>{code}</resultCode><resultMsg>MSG</resultMsg></header>"
        f"<body><items>{body}</items></body></response>"
    ).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    seen_keys: list[str] = []
    bad_key = False

    def log_message(self, *args):  # 시험 출력을 조용히
        pass

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        _Handler.seen_keys.append(query.get("serviceKey", ""))
        op = url.path.rsplit("/", 1)[-1]
        if _Handler.bad_key:
            payload = _xml([], code="30")
        elif op == "getChemList001":
            word = query.get("searchWrd", "")
            if query.get("searchCnd") == "1":
                hits = [(n, c, i) for n, rows in CHEMICALS.items() for c, i in rows if c == word]
            else:
                squash = lambda text: "".join(text.split()).casefold()  # noqa: E731
                hits = [(n, c, i) for n, rows in CHEMICALS.items() for c, i in rows if squash(word) in squash(n) or squash(n) in squash(word)]
            payload = _xml([{"chemId": i, "chemNameKor": n, "casNo": c, "lastDate": "20240101"} for n, c, i in hits])
        else:
            payload = _xml([{"msdsItemNameKor": a, "itemDetail": b} for a, b in DETAIL.get(op, [])])
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def server(monkeypatch):
    _Handler.seen_keys = []
    _Handler.bad_key = False
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", SECRET)
    # probe_msds.main이 모듈 상수를 바꾸므로 시험이 끝나면 원래 값으로 되돌린다
    monkeypatch.setattr(probe_msds.msds_module, "_BASE_URL", probe_msds.msds_module._BASE_URL)
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/B552468/msdschem1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _run(base_url: str, tmp_path, *extra: str) -> int:
    return probe_msds.main(["--base-url", base_url, "--sleep", "0", "--out-dir", str(tmp_path), *extra])


def _rows(tmp_path) -> dict[str, dict]:
    files = list(tmp_path.glob("probe_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    return {r["query"]: r for r in data["rows"]}


def test_build_names_follows_dictionary():
    names = probe_msds.build_names(only_canonical=False, extra=["67-64-1", "톨루엔"], limit=None)
    canonical = [n for n, k in names if k == "표준"]
    assert canonical == list(MATERIALS)
    assert ("크실렌", "별칭") in names
    assert ("67-64-1", "추가") in names
    assert [n for n, _ in names].count("톨루엔") == 1  # 중복 조회 없음
    only = probe_msds.build_names(only_canonical=True, extra=[], limit=3)
    assert len(only) == 3 and all(k == "표준" for _, k in only)
    assert probe_msds.build_names(only_canonical=False, extra=["모호", "67-64-1"], limit=None, skip_dictionary=True) == [
        ("모호", "추가"),
        ("67-64-1", "추가"),
    ]


def test_probe_dictionary_with_aliases(server, tmp_path, capsys):
    assert _run(server, tmp_path) == 0
    files = list(tmp_path.glob("probe_*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    rows = {r["query"]: r for r in data["rows"]}

    assert rows["톨루엔"]["found"] and rows["톨루엔"]["cas_no"] == "108-88-3"
    assert rows["톨루엔"]["api_calls"] == 4  # 목록 1 + 상세 3
    assert rows["톨루엔"]["ghs_has_flammable"] is True
    assert rows["톨루엔"]["regulatory_tag_codes"] == ["CONTROLLED_HAZARDOUS_SUBSTANCE", "DANGEROUS_SUBSTANCE_REGULATION"]
    assert rows["톨루엔"]["result_bytes"] > 0
    assert rows["톨루엔"]["kind"] == "표준"

    # 사전의 표준 표기(자일렌)는 공단 이름(크실렌)과 달라 못 찾고, 별칭(크실렌)으로는 찾는다 (R3)
    assert not rows["자일렌"]["found"] and rows["자일렌"]["reason"] == "후보 0건"
    assert rows["크실렌"]["found"] and rows["크실렌"]["kind"] == "별칭" and rows["크실렌"]["name_equals_query"]

    # 후보는 있지만 이름이 다르면 이유를 말해 준다
    assert not rows["벤젠"]["found"]
    assert rows["벤젠"]["reason"].startswith("이름 불일치")
    assert rows["벤젠"]["candidates"][0] == {"name": "벤젠류", "cas": "71-43-2"}
    # 못 찾은 이유를 보려고 한 목록 호출은 호출 횟수·시간 집계에 넣지 않는다
    assert rows["벤젠"]["api_calls"] == 1
    assert len(rows["벤젠"]["call_seconds"]) == 1

    by_kind = data["summary"]["by_kind"]
    assert by_kind["표준"] == {"total": len(MATERIALS), "found": 3}  # 톨루엔, 아세톤, 이소프로필알코올

    # 공단 이름이 글자까지 같지 않은 것은 이름 매칭(R4)에 걸리므로 공백·대소문자 차이도 드러난다
    assert rows["이소프로필알코올"]["found"] and rows["이소프로필알코올"]["name"] == "이소프로필 알코올"
    assert rows["이소프로필알코올"]["name_equals_query"] is False
    assert "이소프로필알코올" in data["summary"]["name_differs_from_query"]
    assert rows["톨루엔"]["name_equals_query"] is True
    assert by_kind["별칭"]["found"] >= 1
    out = capsys.readouterr().out
    assert "── 요약 ──" in out
    assert "물질 6개 순차 조회 추정" in out


def test_probe_extra_names_only(server, tmp_path):
    assert _run(server, tmp_path, "--no-dictionary", "--names", "모호,없는물질,67-64-1") == 0
    rows = _rows(tmp_path)
    assert list(rows) == ["모호", "없는물질", "67-64-1"]
    assert rows["모호"]["reason"] == "같은 이름에 CAS 2개"
    assert rows["없는물질"]["reason"] == "후보 0건"
    assert rows["67-64-1"]["found"] and rows["67-64-1"]["name"] == "아세톤"
    assert rows["67-64-1"]["name_equals_query"] is False  # 공단 이름이 조회어와 다름 (R4 표시용)


def test_no_dictionary_needs_names(server, tmp_path, capsys):
    assert _run(server, tmp_path, "--no-dictionary") == 2
    assert "--names" in capsys.readouterr().out


def test_limit_must_be_positive(server, tmp_path, capsys):
    assert _run(server, tmp_path, "--limit", "0") == 2
    assert not _Handler.seen_keys  # 연결 확인 전에 거절


def test_report_never_contains_key(server, tmp_path, capsys):
    assert _run(server, tmp_path, "--no-dictionary", "--names", "톨루엔") == 0
    printed = capsys.readouterr()
    saved = next(tmp_path.glob("probe_*.json")).read_text(encoding="utf-8")
    for text in (printed.out, printed.err, saved):
        assert SECRET not in text
    assert SECRET in _Handler.seen_keys[0] or any(SECRET in k for k in _Handler.seen_keys)  # 서버에는 키가 전달됨


def test_ping_only_makes_one_call(server, tmp_path, capsys):
    assert _run(server, tmp_path, "--ping-only") == 0
    assert len(_Handler.seen_keys) == 1
    assert "연결 확인 성공" in capsys.readouterr().out
    assert not list(tmp_path.glob("probe_*.json"))


def test_ping_failure_stops_before_probing(server, tmp_path, capsys):
    _Handler.bad_key = True
    assert _run(server, tmp_path, "--no-dictionary", "--names", "톨루엔") == 1
    out = capsys.readouterr().out
    assert "연결 확인 실패" in out and "resultCode=30" in out
    assert SECRET not in out
    assert len(_Handler.seen_keys) == 1  # 연결 확인 뒤 더 부르지 않는다
    assert not list(tmp_path.glob("probe_*.json"))


def test_encoded_key_warning(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", "abc%2Bdef%3D%3D")
    _run(server, tmp_path, "--ping-only")
    out = capsys.readouterr().out
    assert "Decoding 키" in out
    assert "abc%2Bdef" not in out  # 키 값은 출력하지 않는다


def test_missing_key_exit_code(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("KOSHA_MSDS_SERVICE_KEY", raising=False)
    assert probe_msds.main(["--out-dir", str(tmp_path)]) == 2
    assert "KOSHA_MSDS_SERVICE_KEY" in capsys.readouterr().out


def test_remote_base_url_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", SECRET)
    assert probe_msds.main(["--base-url", "https://example.com/x", "--out-dir", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "127.0.0.1" in out and SECRET not in out
