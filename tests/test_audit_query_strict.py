"""분기 좁히기 v2 (T41 후속): 보고서 분기(E)는 '감사' + 보고서를 만들어 달라는 말일 때만 간다.

- 보고서·자료(명사)만 있고 만들어 달라는 동사가 없으면 기록 질문말이 우선한다.
- E 낱말(이력·기록·지난달 …)만 있는 말도 E로 보내지 않고 기록 질문 안내로 보낸다.
- 이전에 E로 가던 말만 바뀐다. 등록·신청·N시·만들어·작성해와 만들·작성·생성이 든 말은 그대로다.
- 시험은 `.env`·터미널 환경에 기대지 않는다. 모델은 부르지 않는다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.scenario import scenario_policy
from app.agent.scenario.scenario_policy import decide
from app.agent.scenario.scenario_router import RECORD_QUERY_GUIDE, Scenario
from app.core import get_settings

A, E = Scenario.WORK_REGISTER, Scenario.AUDIT_QUERY
ENTRY = (A, E)
ROUTE, GUIDE, WORK, ASK = "보고서 분기(E)", "기록 질문 안내", "작업 등록(A)", "되묻기"


def _kind(d) -> str:
    if d.scenario is E:
        return ROUTE
    if d.scenario is A:
        return WORK
    assert d.scenario is Scenario.UNKNOWN
    return GUIDE if d.redirect is E and d.ask == RECORD_QUERY_GUIDE else ASK


# (문장, v2, STRICT를 끄면(v1))
TABLE = [
    # 바뀌는 말: 이전에 E → 기록 질문 안내
    ("감사 보고서 목록 보여줘", GUIDE, ROUTE),
    ("지난달 실시율 통계 자료 알려줘", GUIDE, ROUTE),
    ("지난달 실시율 보고서 알려줘", GUIDE, ROUTE),
    ("지난주 화기작업 기록", GUIDE, ROUTE),
    ("어제 허가 이력", GUIDE, ROUTE),
    ("지난달 감사 이력", GUIDE, ROUTE),
    ("지난달 감사 대응용 이력", GUIDE, ROUTE),
    ("감사 대응 실시율 보고서", GUIDE, ROUTE),  # 동사가 없어 기록 질문말이 우선 (만들어 달라고 하면 보고서)
    ("지난달 화기작업", GUIDE, ROUTE),  # E 낱말(지난달)만 있는 말
    ("지난달 용접 작업", GUIDE, ROUTE),
    ("감사", GUIDE, ROUTE),
    # 그대로 E: 감사 + 보고서 요청 / 감사 + 보고서·자료 명사(기록 질문말 없음) / 감사 + 대응 / 만들·작성·생성이 든 말
    ("감사 보고서 만들어줘", ROUTE, ROUTE),
    ("감사 보고서 생성해줘", ROUTE, ROUTE),
    ("지난달 감사 자료 작성해줘", ROUTE, ROUTE),
    ("지난달 감사 대응 보고서", ROUTE, ROUTE),
    ("10월 감사 보고서", ROUTE, ROUTE),
    ("감사 자료", ROUTE, ROUTE),
    ("감사 대응", ROUTE, ROUTE),
    ("감사 대응 실시율 보고서 만들어줘", ROUTE, ROUTE),  # 동사가 있으면 보고서
    ("지난달 통계 만들기", ROUTE, ROUTE),  # 감사가 없어도 만들어 달라는 동사가 있으면 이전처럼 E
    # 그대로: 작업 등록·되묻기
    ("내일 용접 작업 등록해줘", WORK, WORK),
    ("오늘 13시 작업", WORK, WORK),
    ("작업 완료 보고서 작성해줘", WORK, WORK),
    ("아차사고 보고서 작성해줘", WORK, WORK),
    ("내일 용접 작업 등록해줘, 필요한 서류 알려줘", WORK, WORK),
    ("감사 보고서 등록해줘", ASK, ASK),
]


@pytest.mark.parametrize("text,after,_before", TABLE, ids=[t[0] for t in TABLE])
def test_sentence_table_v2(text, after, _before):
    assert _kind(decide(text, mode="VERIFY", allowed=ENTRY)) == after


@pytest.mark.parametrize("text,_after,before", TABLE, ids=[t[0] for t in TABLE])
def test_strict_off_restores_v1(monkeypatch, text, _after, before):
    monkeypatch.setattr(scenario_policy, "AUDIT_QUERY_STRICT", False)
    assert _kind(decide(text, mode="VERIFY", allowed=ENTRY)) == before


def test_split_off_also_turns_strict_off(monkeypatch):
    monkeypatch.setattr(scenario_policy, "AUDIT_QUERY_SPLIT", False)
    assert decide("지난주 화기작업 기록", mode="VERIFY", allowed=ENTRY).scenario is E


def test_only_sentences_that_went_to_e_change():
    # v1 → v2에서 바뀌는 말은 전부 이전에 E였고 기록 질문 안내로 간다. 다른 쪽으로 가던 말은 안 바뀐다.
    changed = [(t, a, b) for t, a, b in TABLE if a != b]
    assert changed and all(a == GUIDE and b == ROUTE for _, a, b in changed)


# ── 안전 조건 ──


@pytest.mark.parametrize(
    "text",
    ["내일 용접 작업 등록, 어제 기록 알려줘", "어제 하던 용접 이어서 내일 9시에 알려줘", "어제 기록 만들어줘 등록", "지난달 기록 작성해줘"],
)
def test_register_clock_or_make_words_never_become_a_guide(text):
    assert decide(text, mode="VERIFY", allowed=ENTRY).ask != RECORD_QUERY_GUIDE


def test_e_outside_chat_scope_keeps_the_old_redirect():
    # E가 채팅 범위 밖이면 안내하지 않고 이전처럼 범위 밖 안내 (v1과 같음)
    for text in ("지난달 화기작업", "지난주 화기작업 기록", "감사 보고서 목록 보여줘"):
        d = decide(text, mode="VERIFY", allowed=(A,))
        assert d.scenario is Scenario.UNKNOWN and d.redirect is E and "전용 화면" in d.ask, text


def test_why_not_ask_back_the_answer_is_joined_with_the_first_message():
    # 되묻기에 대한 답은 앞선 말과 이어 붙여 다시 판단된다. 그래서 E 낱말만 있는 말을 되물으면
    # '지난달 용접 작업' + '작업 등록'이 다시 되묻기가 되어 되풀이한다. 안내로 보내는 까닭이다.
    d = decide("지난달 용접 작업\\n작업 등록", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.redirect is None


# ── work-register 응답 ──


def _client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "CHECK_RULES_TOOL": "MOCK",
        "MSDS_TOOL": "MOCK",
        "NETWORK_ENV": "OPEN",
        "LLM_PRELOAD": "false",
        "RAG_ENABLED": "false",
        "SCENARIO_ENGINES": '{"default": {"engine": "FAKE"}}',
        "AUDIT_ROUTE_ENABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("KOSHA_MSDS_SERVICE_KEY", raising=False)
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def _post(client, message, slots=None):
    body = {"runId": 1, "siteId": 1, "mode": "VERIFY", "message": message, "slots": slots}
    response = client.post("/api/v1/agent/work-register", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize(
    "text", ["지난주 화기작업 기록", "감사 보고서 목록 보여줘", "지난달 실시율 통계 자료 알려줘", "지난달 용접 작업"]
)
def test_record_questions_with_e_words_get_the_guide(monkeypatch, tmp_path, text):
    data = _post(_client(monkeypatch, tmp_path), text)
    assert data["runStatus"] == "WAITING_INPUT" and data["route"] is None and data["drafts"] is None
    assert data["ask"]["message"] == RECORD_QUERY_GUIDE and data["slots"]["pendingText"] is None


def test_after_the_guide_the_next_message_is_judged_on_its_own(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    first = _post(client, "지난달 용접 작업")
    second = _post(client, "지난달 감사 대응 보고서 만들어줘", slots=first["slots"])
    assert second["runStatus"] == "DONE" and second["route"]["target"] == "AUDIT_REPORT"
