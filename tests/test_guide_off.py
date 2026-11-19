"""F(소규모 사업장 안내) 제외: 채팅 범위 A·E, mode=GUIDE 요청도 A로 처리한다.

시험은 `.env`·터미널 환경에 기대지 않는다. 모델은 부르지 않는다 (SCENARIO_ENGINES default=FAKE, 시나리오 판단은 키워드 규칙,
계획은 모델 응답을 못 읽어 규칙 기반 계획으로 간다).
"""

import pytest
from fastapi.testclient import TestClient

from app.agent.scenario.scenario_dispatch import ScenarioDispatcher
from app.agent.scenario.scenario_handlers import load_chat_scenarios, register_work_register_handlers
from app.agent.scenario.scenario_policy import decide
from app.agent.scenario.scenario_router import Scenario
from app.core import get_settings
from app.llm.llm_provider import LLMClientProvider

A, E, F = Scenario.WORK_REGISTER, Scenario.AUDIT_QUERY, Scenario.GUIDE
WORK = "내일 오후 2시부터 5시까지 3층 용접 작업 등록해줘, 인원 4명, 주변 물질 없음, 장비는 용접기"
PERIOD = "2026-10-01~2026-10-31"


def _client(monkeypatch, tmp_path, *, chat=None, route=False):
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않게
    for key, value in {
        "CHECK_RULES_TOOL": "MOCK",
        "MSDS_TOOL": "MOCK",
        "NETWORK_ENV": "OPEN",
        "LLM_PRELOAD": "false",
        "RAG_ENABLED": "false",
        "SCENARIO_ENGINES": '{"default": {"engine": "FAKE"}}',
        "AUDIT_ROUTE_ENABLED": "true" if route else "false",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KOSHA_MSDS_SERVICE_KEY", raising=False)
    if chat is None:
        monkeypatch.delenv("CHAT_SCENARIOS", raising=False)
    else:
        monkeypatch.setenv("CHAT_SCENARIOS", chat)
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def _post(client, message, mode, slots=None):
    body = {"runId": 1, "siteId": 1, "mode": mode, "message": message, "slots": slots}
    response = client.post("/api/v1/agent/work-register", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


# ── 설정 ──────────────────────────────────────────────────────


def test_default_chat_scenarios_has_no_f(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHAT_SCENARIOS", raising=False)
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")
    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("MSDS_TOOL", "MOCK")
    get_settings.cache_clear()
    assert get_settings().chat_scenarios == "A,E"


@pytest.mark.parametrize("raw, expected", [("A,E", (A, E)), ("A", (A,)), ("A,E,F", (A, E, F)), ("E", (E,)), (None, None)])
def test_load_chat_scenarios_accepts(raw, expected):
    assert load_chat_scenarios(raw) == expected


@pytest.mark.parametrize("raw", ["F", "E,F"])
def test_f_without_a_is_still_refused(raw):
    with pytest.raises(ValueError, match="F를 넣으려면 A도"):
        load_chat_scenarios(raw)


# ── 시나리오 판단 ─────────────────────────────────────────────


def test_guide_mode_not_forced_when_f_is_out_of_scope():
    assert decide("감사 보고서 만들어줘", mode="GUIDE", allowed=(A, E)).scenario is E
    assert decide("내일 용접 작업 등록해줘", mode="GUIDE", allowed=(A, E)).scenario is A


@pytest.mark.parametrize("allowed", [None, (A, E, F)])
def test_guide_mode_still_forced_to_f_when_f_is_in_scope(allowed):
    assert decide("감사 보고서 만들어줘", mode="GUIDE", allowed=allowed).scenario is F


# ── 처리기 등록 ───────────────────────────────────────────────


class _Provider:
    """처리기 등록만 보는 가짜 (모델은 만들지 않는다)."""


def _registered(guide: bool):
    dispatcher = ScenarioDispatcher(_Provider())  # type: ignore[arg-type]
    register_work_register_handlers(dispatcher, tools=None, guide=guide)  # type: ignore[arg-type]
    return dispatcher.available_scenarios


def test_register_without_guide_has_only_a():
    assert _registered(False) == (A,)


def test_register_default_keeps_a_and_f():
    assert set(_registered(True)) == {A, F}


# ── work-register 응답 ────────────────────────────────────────


def test_guide_mode_work_is_handled_by_a_and_keeps_easy_words(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path), WORK, "GUIDE")
    assert data["runStatus"] == "DONE"
    assert data["ask"]["message"].startswith("오늘 작업에 필요한 서류 초안을")  # 쉬운 말 완료 문구 = mode=GUIDE가 에이전트에 전달됨
    assert data["guidance"], "GUIDE 안내문(guidance)이 남아야 한다"


def test_verify_mode_unchanged(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path), WORK, "VERIFY")
    assert data["runStatus"] == "DONE"
    assert data["ask"]["message"].startswith("초안을 만들었습니다.")
    assert data["guidance"] == []


def test_guide_mode_can_continue_a_conversation(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    first = _post(client, "용접 작업 하려고요", "GUIDE")
    assert first["runStatus"] == "WAITING_INPUT" and first["slots"]["utterances"]
    second = _post(client, "내일 오후 2시부터 5시까지 3층, 4명, 주변 물질 없음, 장비는 용접기", "GUIDE", slots=first["slots"])
    assert second["runStatus"] == "DONE"


def test_guide_mode_audit_sentence_goes_to_audit_route(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path, route=True), f"{PERIOD} 감사 보고서 만들어줘", "GUIDE")
    assert data["runStatus"] == "DONE"
    assert data["route"]["target"] == "AUDIT_REPORT"


def test_f_in_chat_scenarios_restores_old_behaviour(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path, chat="A,E,F", route=True)
    # GUIDE 현장은 말을 보지 않고 F로 간다: 감사 문장도 감사 대응으로 가지 않는다
    data = _post(client, f"{PERIOD} 감사 보고서 만들어줘", "GUIDE")
    assert data["route"] is None and data["runStatus"] == "WAITING_INPUT"
    # F 처리기로 작업 등록이 돈다 (쉬운 말 안내문)
    work = _post(client, WORK, "GUIDE")
    assert work["runStatus"] == "DONE" and work["guidance"]
    # VERIFY 현장은 그대로 A·E
    assert _post(client, f"{PERIOD} 감사 보고서 만들어줘", "VERIFY")["route"]["target"] == "AUDIT_REPORT"
