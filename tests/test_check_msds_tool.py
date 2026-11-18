"""scripts/check_msds_tool.py 시험 (공단 API 호출 없음, 설정은 환경변수로)."""

from __future__ import annotations

import pytest

from app.agent import agent_deps
from app.core import get_settings
from scripts import check_msds_tool

TOLUENE = "인화성 액체 : 구분2|피부 부식성/피부 자극성 : 구분2|생식독성 : 구분2|특정표적장기 독성(1회 노출) : 구분3(마취영향)|흡인 유해성 : 구분1"


def _fake_request_items(operation, parameters):
    """공단 API 응답 흉내 (실제 모양: 목록 → 유해성·응급조치·규제)."""
    if operation == "getChemList001":
        table = {"메틸알코올": ("메틸 알코올", "67-56-1", "C1"), "이소프로필알코올": ("이소프로필 알코올", "67-63-0", "C2")}
        if parameters["searchWrd"] in table:
            name, cas, chem_id = table[parameters["searchWrd"]]
            return [{"chemId": chem_id, "chemNameKor": name, "casNo": cas, "lastDate": "20240101"}]
        return []
    if operation == "getChemDetail021":
        return [{"msdsItemNameKor": "유해성·위험성 분류", "itemDetail": TOLUENE}, {"msdsItemNameKor": "유해성·위험성 분류 2", "itemDetail": "자료없음"}]
    if operation == "getChemDetail041":
        return [{"msdsItemNameKor": "흡입했을 때", "itemDetail": "신선한 공기로 옮긴다."}]
    return [{"msdsItemNameKor": "산업안전보건법에 의한 규제", "itemDetail": "관리대상유해물질"}]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for name in ("KOSHA_MSDS_SERVICE_KEY", "INTERNAL_API_BASE_URL", "INTERNAL_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    # 이 시험은 lookup_msds만 본다. RAG를 켜 두면(기본) 도구 등록이 검색 백엔드를 만들다 EMBEDDING_URL이 없는 환경에서
    # 설정 오류(종료 코드 2)가 난다 (11/17 PC에서 3건 실패).
    monkeypatch.setenv("RAG_ENABLED", "false")
    monkeypatch.setenv("MSDS_TOOL", "MOCK")  # 코드 기본값은 KOSHA(키 필요). 이 시험들은 MOCK에서 시작하고 KOSHA는 시험 안에서 켠다
    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않게
    get_settings.cache_clear()
    agent_deps.get_tool_registry.cache_clear()
    yield
    get_settings.cache_clear()
    agent_deps.get_tool_registry.cache_clear()


def test_mock_is_reported_and_require_real_fails(capsys):
    assert check_msds_tool.main(["톨루엔"]) == 0
    out = capsys.readouterr().out
    assert "mock (연습 데이터)" in out and "[mock]" in out
    assert check_msds_tool.main(["--require-real"]) == 1
    assert "실제 도구가 아닙니다" in capsys.readouterr().out


def test_kosha_is_reported_with_official_names(monkeypatch, capsys):
    from app.publicdata.msds import KoshaMsdsClient

    monkeypatch.setenv("MSDS_TOOL", "KOSHA")
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", "TEST-KEY-123")
    monkeypatch.setattr(KoshaMsdsClient, "_request_items", lambda self, operation, parameters: _fake_request_items(operation, parameters))
    assert check_msds_tool.main(["--require-real", "메탄올", "이소프로필알코올", "시너"]) == 0
    out = capsys.readouterr().out
    assert "실제 (공단 MSDS API)" in out
    assert "메탄올: 찾음 이름 '메틸 알코올' CAS 67-56-1" in out
    assert "이소프로필알코올: 찾음 이름 '이소프로필 알코올'" in out
    assert "ghs ['인화성 액체'" in out  # 덩어리가 아니라 라벨 목록
    assert "시너: 못 찾음" in out
    assert "응급조치 첫 줄: 흡입했을 때: 신선한 공기로 옮긴다." in out
    assert "TEST-KEY-123" not in out


def test_kosha_without_key_is_a_settings_error(monkeypatch, capsys):
    monkeypatch.setenv("MSDS_TOOL", "KOSHA")
    assert check_msds_tool.main([]) == 2
    out = capsys.readouterr().out
    assert "설정 오류" in out and "KOSHA_MSDS_SERVICE_KEY" in out


def test_all_failed_is_reported(monkeypatch, capsys):
    from app.exceptions import ToolError
    from app.publicdata.msds import KoshaMsdsClient

    def boom(self, operation, parameters):
        raise ToolError("MSDS_UPSTREAM_ERROR", "공단 MSDS API에 연결하거나 응답을 처리하지 못했습니다.", tool="lookup_msds")

    monkeypatch.setenv("MSDS_TOOL", "KOSHA")
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", "TEST-KEY-123")
    monkeypatch.setattr(KoshaMsdsClient, "_request_items", boom)
    assert check_msds_tool.main(["톨루엔"]) == 1
    out = capsys.readouterr().out
    assert "조회 실패: MSDS_UPSTREAM_ERROR" in out and "TEST-KEY-123" not in out
