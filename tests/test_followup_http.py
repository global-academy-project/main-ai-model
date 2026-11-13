"""followup을 `/api/v1/agent/work-register` 라우트까지 거쳐 확인한다 (스텁 하니스: 가짜 LLM + mock 도구).

test_followup_cycle.py는 `WorkRegisterAgent.run`을 직접 부른다. 여기서는 실제 앱(`create_app`)의 라우트, 디스패처
(이어 가기 확인), 응답 직렬화를 거친다. Spring이 하는 일은 흉내 낸다: 직전 응답의 `data.slots`(camelCase JSON)를
저장했다가 다음 요청의 `slots`로 그대로 돌려준다 (null 필드가 빠지는 경우도 본다).

환경에 의존하지 않는다 (T17): 설정 환경변수를 직접 정하고 `.env`가 없는 폴더에서 돌린다.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agent import agent_deps, followup
from app.core import get_settings
from app.llm.llm_fake import FakeLLMClient, text_response
from app.llm.llm_provider import LLMClientProvider
from app.tools.tool_setup import build_tool_registry

ROUTE = "/api/v1/agent/work-register"
PLAN = json.dumps(
    {"summary": "R-201 반응기 하부 배관 플랜지 볼트 교체, 고착 시 그라인더 절단", "equipment": ["그라인더"], "materials": ["톨루엔"]},
    ensure_ascii=False,
)
FIRST = "오늘 13시에 화기작업 있어. 필요한 거 만들어줘."
DETAIL = "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 옆 T-105 탱크에 톨루엔. 3명, 15시까지. 크레인은 안 씀."
NEW_TASK = "T-201 탱크 내부 청소하는 밀폐공간 작업 있어"

# 이 시험이 읽는 설정. 환경(.env, 컨테이너 변수)에 있는 값이 시험을 바꾸지 않게 모두 비운다.
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
    app.dependency_overrides[agent_deps.get_tool_registry] = lambda: build_tool_registry()  # mock 도구 (RAG·Spring 없음)
    client = TestClient(app, raise_server_exceptions=False)
    client.fake = fake  # type: ignore[attr-defined]
    yield client
    get_settings.cache_clear()


def strip_nulls(value):
    """Spring이 null 필드를 빼고 저장해도 같게 동작하는지 보려고 쓴다."""
    if isinstance(value, dict):
        return {k: strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_nulls(v) for v in value]
    return value


def turn(client, message, slots=None, mode="VERIFY"):
    """Spring 한 번: POST하고 data를 돌려준다 (직전 data.slots를 slots로 넘긴다)."""
    body = {"runId": 1, "siteId": 1, "message": message, "slots": slots, "mode": mode}
    response = client.post(ROUTE, json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True, payload
    return payload["data"]


def step_names(data):
    return [s["name"] for s in data["steps"]]


@pytest.fixture
def done(api):
    first = turn(api, FIRST)
    second = turn(api, DETAIL, first["slots"])
    assert first["runStatus"] == "WAITING_INPUT" and second["runStatus"] == "DONE"
    return api, second


def test_route_first_two_turns_reach_done(done):
    _, second = done
    assert second["slots"]["headcount"] == 3 and second["slots"]["utterances"]
    assert second["ask"]["missingFields"] == [] and second["classification"]["workTypes"] == ["화기"]


@pytest.mark.parametrize("message", ["고마워", "위험요인 알려줘"])
def test_route_unchanged_message_guides_without_pipeline(done, message):
    client, previous = done
    calls = len(client.fake.calls)
    data = turn(client, message, previous["slots"])
    assert data["runStatus"] == "WAITING_INPUT" and data["ask"]["missingFields"] == []
    assert data["ask"]["message"] == followup.REPLY_UNCHANGED
    assert data["slots"] == previous["slots"]  # 직렬화를 거쳐도 slots가 그대로다
    assert step_names(data) == ["classify_work"] and len(client.fake.calls) == calls  # 계획 LLM 호출 없음


def test_route_edit_regenerates_with_change_note(done):
    client, previous = done
    data = turn(client, "인원 5명으로 바꿔줘", previous["slots"])
    assert data["runStatus"] == "DONE" and data["ask"]["message"].startswith("변경: 인원 3명 → 5명")
    assert data["slots"]["headcount"] == 5 and len(client.fake.calls) > 0


def test_route_replace_expression_swaps_equipment(done):
    client, previous = done
    data = turn(client, "그라인더 말고 용접으로 해줘", previous["slots"])
    assert data["runStatus"] == "DONE" and data["ask"]["message"].startswith("변경: ")
    assert "용접" in (data["slots"]["hotWorkKind"] or "")


def test_route_new_task_guides_and_keeps_previous_draft(done):
    client, previous = done
    calls = len(client.fake.calls)
    data = turn(client, NEW_TASK, previous["slots"])
    assert data["runStatus"] == "WAITING_INPUT" and data["ask"]["message"] == followup.REPLY_NEW_TASK
    assert "새 대화" in data["ask"]["message"] and data["slots"] == previous["slots"]
    assert step_names(data) == ["classify_work"] and len(client.fake.calls) == calls
    again = turn(client, "인원 5명으로 바꿔줘", data["slots"])  # 안내 뒤에도 이전 초안을 고칠 수 있다
    assert again["runStatus"] == "DONE" and again["slots"]["headcount"] == 5


def test_route_works_when_spring_drops_null_fields(api):
    first = turn(api, FIRST)
    second = turn(api, DETAIL, strip_nulls(first["slots"]))
    assert second["runStatus"] == "DONE"
    guided = turn(api, "고마워", strip_nulls(second["slots"]))
    assert guided["ask"]["message"] == followup.REPLY_UNCHANGED
    edited = turn(api, "16시까지로 바꿔줘", strip_nulls(guided["slots"]))
    assert edited["runStatus"] == "DONE" and edited["ask"]["message"].startswith("변경: ")


def test_route_guide_mode_site_also_uses_followup(api):
    first = turn(api, FIRST, mode="GUIDE")
    second = turn(api, DETAIL, first["slots"], mode="GUIDE")
    assert second["runStatus"] == "DONE"
    data = turn(api, "고마워", second["slots"], mode="GUIDE")
    assert data["runStatus"] == "WAITING_INPUT" and data["ask"]["message"] == followup.REPLY_UNCHANGED


def test_route_followup_disabled_reruns_like_before(done, monkeypatch):
    monkeypatch.setattr(followup, "FOLLOWUP_ENABLED", False)
    client, previous = done
    data = turn(client, "고마워", previous["slots"])
    assert data["runStatus"] == "DONE" and not data["ask"]["message"].startswith(followup.REPLY_UNCHANGED)
    assert "evidence.plan" in step_names(data)  # 변경 전처럼 파이프라인 전체가 돈다


def test_route_state_comes_only_from_slots(api):
    """Spring이 slots를 안 돌려주면 (대화를 닫은 경우) 후속 처리를 하지 않고 새 대화처럼 분류한다."""
    first = turn(api, FIRST)
    second = turn(api, DETAIL, first["slots"])
    assert second["runStatus"] == "DONE"
    data = turn(api, "고마워", None)
    assert data["ask"]["message"] != followup.REPLY_UNCHANGED  # followup이 끼어들지 않는다
