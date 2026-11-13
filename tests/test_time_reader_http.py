"""시각 읽기 개선을 `/api/v1/agent/work-register` 라우트까지 거쳐 확인한다 (스텁 하니스: 가짜 LLM + mock 도구).

Spring이 하는 일은 흉내 낸다: 직전 응답의 `data.slots`(camelCase JSON)를 저장했다가 다음 요청의 `slots`로 그대로 돌려준다
(null 필드가 빠지는 경우도 본다). 환경에 의존하지 않는다 (T17): 설정 환경변수를 직접 정하고 `.env`가 없는 폴더에서 돌린다.
DONE 뒤 후속 말(followup) 시험은 followup 패치가 적용된 저장소에서만 돈다.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import agent_deps, time_reader
from app.core import get_settings
from app.llm.llm_fake import FakeLLMClient, text_response
from app.llm.llm_provider import LLMClientProvider
from app.tools.tool_setup import build_tool_registry

ROUTE = "/api/v1/agent/work-register"
PLAN = json.dumps(
    {"summary": "R-201 반응기 하부 배관 플랜지 볼트 교체, 고착 시 그라인더 절단", "equipment": ["그라인더"], "materials": ["톨루엔"]},
    ensure_ascii=False,
)
# 시간만 빼고 필수 항목이 다 있는 말 (되묻기 항목이 시작·종료뿐이다. 화기작업이라 LEL 값도 말해 둔다)
BASE = "R-201 반응기 하부 배관 용접 작업. 옆 T-105 탱크에 톨루엔. 3명. 그라인더 사용. LEL 0%."
HAS_FOLLOWUP = importlib.util.find_spec("app.agent.followup") is not None
needs_followup = pytest.mark.skipif(not HAS_FOLLOWUP, reason="followup 패치가 없는 저장소")

CLEARED = (
    "CHAT_SCENARIOS", "SCENARIO_ENGINES", "LLM_ADAPTERS", "INTERNAL_API_BASE_URL", "INTERNAL_API_TOKEN",
    "OPENAI_API_KEY", "OPENAI_MODEL", "AI_MAX_OUTPUT_TOKENS", "AI_TIMEOUT_SECONDS", "RAG_BACKEND",
    "KOSHA_MSDS_SERVICE_KEY",
)


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # .env를 읽지 않는다
    for name in CLEARED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")
    monkeypatch.setenv("RAG_ENABLED", "false")
    get_settings.cache_clear()
    from app.main import create_app  # 가져오는 순간 설정을 읽으므로 환경변수를 정한 뒤에

    app = create_app()
    fake = FakeLLMClient([text_response(PLAN)] * 60)
    app.state.llm_provider = LLMClientProvider(get_settings(), {}, factory=lambda engine, model: fake)
    app.dependency_overrides[agent_deps.get_tool_registry] = lambda: build_tool_registry()
    client = TestClient(app, raise_server_exceptions=False)
    client.fake = fake  # type: ignore[attr-defined]
    yield client
    get_settings.cache_clear()


def strip_nulls(value):
    if isinstance(value, dict):
        return {k: strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_nulls(v) for v in value]
    return value


def turn(client, message, slots=None, mode="VERIFY"):
    body = {"runId": 1, "siteId": 1, "message": message, "slots": slots, "mode": mode}
    response = client.post(ROUTE, json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True, payload
    return payload["data"]


def time_keys(data):
    return [f["key"] for f in data["ask"]["missingFields"]]


def test_route_unreadable_time_asks_with_reason_and_example(api):
    data = turn(api, f"{BASE} 9시 11시 14시")
    assert data["runStatus"] == "WAITING_INPUT" and time_keys(data) == ["start_time", "end_time"]
    message = data["ask"]["message"]
    assert time_reader.PREFIX_UNREAD.strip() in message and "9시부터 11시까지" in message
    assert message.count("9시부터 11시까지") == 1  # 시작·종료 문구가 같아도 한 줄이다
    assert data["slots"]["startTime"] is None and data["slots"]["endTime"] is None


def test_route_label_answer_ends_the_question_loop(api):
    first = turn(api, f"{BASE} 9시 11시 14시")
    second = turn(api, "시작은 9시, 종료는 11시", first["slots"])
    assert second["runStatus"] == "DONE" and second["slots"]["startTime"] == "09:00" and second["slots"]["endTime"] == "11:00"


def test_route_half_answer_then_other_half(api):
    first = turn(api, BASE)
    assert time_keys(first) == ["start_time", "end_time"] and time_reader.Q_BOTH in first["ask"]["message"]
    second = turn(api, "종료는 11시", first["slots"])
    assert second["runStatus"] == "WAITING_INPUT" and time_keys(second) == ["start_time"]
    assert time_reader.Q_START in second["ask"]["message"] and second["slots"]["endTime"] == "11:00"
    third = turn(api, "시작은 9시", strip_nulls(second["slots"]))  # Spring이 null을 빼도 같다
    assert third["runStatus"] == "DONE" and third["slots"]["startTime"] == "09:00"


@pytest.mark.parametrize("phrase", ["15시부터 11시까지", "9시부터 9시까지", "오후 2시부터 오전 9시까지"])
def test_route_mismatch_blanks_both_and_guides(api, phrase):
    data = turn(api, f"{BASE} {phrase}")
    assert data["runStatus"] == "WAITING_INPUT" and time_keys(data) == ["start_time", "end_time"]
    assert time_reader.MSG_MISMATCH in data["ask"]["message"]
    assert data["slots"]["startTime"] is None and data["slots"]["endTime"] is None
    fixed = turn(api, "오전 9시부터 오후 2시까지", data["slots"])
    assert fixed["runStatus"] == "DONE" and (fixed["slots"]["startTime"], fixed["slots"]["endTime"]) == ("09:00", "14:00")


@pytest.mark.parametrize("phrase", ["23시부터 2시까지", "밤 10시부터 새벽 2시까지"])
def test_route_overnight_is_guided_not_registered(api, phrase):
    data = turn(api, f"{BASE} {phrase}")
    assert data["runStatus"] == "WAITING_INPUT" and time_reader.MSG_OVERNIGHT in data["ask"]["message"]
    assert data["slots"]["startTime"] is None and data["slots"]["endTime"] is None
    assert [s["name"] for s in data["steps"]] == ["classify_work"]  # 되묻는 턴은 classify_work만 부른다
    fixed = turn(api, "19시부터 23시까지", data["slots"])
    assert fixed["runStatus"] == "DONE" and fixed["slots"]["endTime"] == "23:00"


def test_route_misread_phrases_are_not_registered_wrongly(api):
    first = turn(api, f"{BASE} 11시에 마쳐")  # 예전에는 시작 11:00으로 읽었다
    assert first["slots"]["startTime"] is None and first["slots"]["endTime"] == "11:00"
    assert time_keys(first) == ["start_time"]


@pytest.fixture
def done(api):
    first = turn(api, "오늘 13시에 화기작업 있어. 필요한 거 만들어줘.")
    second = turn(
        api,
        "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 옆 T-105 탱크에 톨루엔. 3명, 15시까지. 크레인은 안 씀.",
        first["slots"],
    )
    assert second["runStatus"] == "DONE"
    return api, second


def test_route_first_two_turns_reach_done(done):
    _, second = done
    assert second["slots"]["startTime"] == "13:00" and second["slots"]["endTime"] == "15:00"


@needs_followup
def test_route_done_then_labelled_edit(done):
    client, previous = done
    data = turn(client, "종료 시간을 16시로 바꿔줘", previous["slots"])
    assert data["runStatus"] == "DONE" and data["ask"]["message"].startswith("변경: 종료 15:00 → 16:00")
    assert data["slots"]["endTime"] == "16:00" and data["slots"]["startTime"] == "13:00"
    again = turn(client, "15시까지가 아니라 17시까지로 바꿔줘", data["slots"])
    assert again["runStatus"] == "DONE" and again["slots"]["endTime"] == "17:00"


@needs_followup
def test_route_done_then_role_unknown_edit_guides_without_change(done):
    client, previous = done
    calls = len(client.fake.calls)
    data = turn(client, "16시로 변경", previous["slots"])
    assert data["runStatus"] == "WAITING_INPUT" and data["ask"]["message"] == time_reader.REPLY_TIME_ROLE
    assert data["slots"] == previous["slots"] and len(client.fake.calls) == calls  # 값도 계획 LLM 호출도 없다


@needs_followup
def test_route_done_then_conflicting_edit_blanks_both_and_asks(done):
    client, previous = done
    data = turn(client, "11시까지로 바꿔줘", previous["slots"])
    assert data["runStatus"] == "WAITING_INPUT" and time_keys(data) == ["start_time", "end_time"]
    assert time_reader.MSG_MISMATCH in data["ask"]["message"]
    assert data["slots"]["startTime"] is None and data["slots"]["endTime"] is None
    fixed = turn(client, "13시부터 16시까지", data["slots"])
    assert fixed["runStatus"] == "DONE" and fixed["slots"]["endTime"] == "16:00"
