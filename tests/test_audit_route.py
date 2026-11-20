"""채팅 감사 대응 분기: 규칙(scenario_policy) → E 처리기 → work-register 응답의 route.

시험은 `.env`·터미널 환경에 기대지 않는다: 필요한 환경변수를 시험 안에서 정하고 설정 캐시를 비운다.
모델은 부르지 않는다 (SCENARIO_ENGINES={"default":"FAKE"}, 시나리오 판단은 키워드 규칙).
"""

import pytest
from fastapi.testclient import TestClient

from app.agent.scenario import scenario_policy
from app.agent.scenario.scenario_policy import decide
from app.agent.scenario.scenario_router import Scenario
from app.core import get_settings

PERIOD = "2026-10-01~2026-10-31"


def _client(monkeypatch, tmp_path, *, route: bool | None):
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않게
    for key, value in {
        "CHECK_RULES_TOOL": "MOCK",
        "MSDS_TOOL": "MOCK",
        "NETWORK_ENV": "OPEN",
        "LLM_PRELOAD": "false",
        "RAG_ENABLED": "false",
        "SCENARIO_ENGINES": '{"default": {"engine": "FAKE"}}',
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KOSHA_MSDS_SERVICE_KEY", raising=False)
    if route is None:
        monkeypatch.delenv("AUDIT_ROUTE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AUDIT_ROUTE_ENABLED", "true" if route else "false")
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def _post(client, message, slots=None):
    body = {"runId": 1, "siteId": 1, "mode": "VERIFY", "message": message, "slots": slots}
    response = client.post("/api/v1/agent/work-register", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


# ── 시나리오 판단 규칙 ─────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["감사 보고서 만들어줘", "지난달 감사 자료 작성해줘", "감사 대응 보고서 만들어 줘", "10월 감사 보고서"],  # '지난달 실시율 알려줘'는 기록 질문 안내 (test_audit_query_split.py)
)
def test_audit_sentences_go_to_e(text):
    assert decide(text, mode="VERIFY").scenario is Scenario.AUDIT_QUERY


@pytest.mark.parametrize(
    "text",
    ["내일 오후 2시에 3층 용접 작업 등록해줘", "감사합니다 내일 용접 작업 등록해줘", "감사합니다. 용접 작업 하려고요"],
)
def test_work_sentences_stay_a(text):
    assert decide(text, mode="VERIFY").scenario is Scenario.WORK_REGISTER


def test_register_word_with_audit_still_asks():
    # '등록'이 함께 있으면 이전 규칙 그대로: 정하지 않고 되묻는다
    assert decide("감사 보고서 등록해줘", mode="VERIFY").scenario is Scenario.UNKNOWN


def test_audit_priority_switch_off_restores_old_rule(monkeypatch):
    monkeypatch.setattr(scenario_policy, "AUDIT_PRIORITY", False)
    assert decide("감사 보고서 만들어줘", mode="VERIFY").scenario is Scenario.UNKNOWN


def test_audit_priority_switch_off_reads_thanks_as_audit_keyword(monkeypatch):
    # 이전 규칙: 인사말 '감사합니다'도 E 키워드('감사')로 읽혔다
    monkeypatch.setattr(scenario_policy, "AUDIT_PRIORITY", False)
    assert decide("감사합니다. 용접 작업 하려고요", mode="VERIFY").scenario is Scenario.AUDIT_QUERY


def test_guide_mode_is_always_f():
    assert decide("감사 보고서 만들어줘", mode="GUIDE").scenario is Scenario.GUIDE


# ── work-register 응답 ────────────────────────────────────────


def test_period_in_message_returns_route(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path, route=True), f"{PERIOD} 감사 보고서 만들어줘")
    assert data["runStatus"] == "DONE"
    assert data["route"]["target"] == "AUDIT_REPORT"
    assert data["route"]["dateFrom"] == "2026-10-01"
    assert data["route"]["dateTo"] == "2026-10-31"
    assert "2026-10-01 ~ 2026-10-31" in data["route"]["message"]
    assert data["drafts"] is None and data["ask"] is None and data["steps"] == []


def test_missing_period_asks_and_next_answer_routes(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, route=True)
    first = _post(client, "감사 보고서 만들어줘")
    assert first["runStatus"] == "WAITING_INPUT"
    assert first["route"] is None
    assert first["ask"]["missingFields"][0]["key"] == "audit_period"
    assert first["slots"]["pendingText"] == "감사 보고서 만들어줘"
    # 되묻기 응답의 slots를 그대로 돌려보내고, 기간만 답한다 (앞선 말과 이어 붙어 다시 분류된다)
    second = _post(client, PERIOD, slots=first["slots"])
    assert second["runStatus"] == "DONE"
    assert second["route"]["dateFrom"] == "2026-10-01"
    assert second["route"]["dateTo"] == "2026-10-31"
    assert second["slots"]["pendingText"] is None


def test_ambiguous_period_asks_for_one(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path, route=True), "지난달과 이번 달 감사 보고서")
    assert data["runStatus"] == "WAITING_INPUT" and data["route"] is None
    assert "하나만" in data["ask"]["message"]


@pytest.mark.parametrize("route", [False])
def test_switch_off_keeps_old_behaviour(monkeypatch, tmp_path, route):
    data = _post(_client(monkeypatch, tmp_path, route=route), f"{PERIOD} 감사 보고서 만들어줘")
    assert data["runStatus"] == "WAITING_INPUT"
    assert data["route"] is None
    assert "전용 화면" in data["ask"]["message"]
    assert data["slots"]["pendingText"] is None


def test_default_follows_the_setting_default(monkeypatch, tmp_path):
    """환경변수가 없으면 Settings 기본값을 따른다 (코드 기본값을 바꿔도 시험이 따라간다)."""
    from app.core import Settings

    default_on = Settings.model_fields["audit_route_enabled"].default
    data = _post(_client(monkeypatch, tmp_path, route=None), f"{PERIOD} 감사 보고서 만들어줘")
    if default_on:
        assert data["runStatus"] == "DONE" and data["route"] is not None
    else:
        assert data["runStatus"] == "WAITING_INPUT" and data["route"] is None
