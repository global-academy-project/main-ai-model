"""기록 질문 분리 (C 요청 2-3, T41): 규칙 · 안내 문구 · 스위치 · work-register 응답.

- 감사 대응 보고서를 만들어 달라는 말만 E(보고서 분기)로 가고, 기록을 묻는 말은 기록 질문 탭 안내로 간다.
- 시험은 `.env`·터미널 환경에 기대지 않는다. 모델은 부르지 않는다 (시나리오 판단은 키워드 규칙).
- 문장표는 `감사대응_기록질문_분기_좁히기_설계_v0.md` 2-3장과 같다 (1~14번은 C 표, 15~19번은 B가 더한 문장).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.scenario import scenario_policy
from app.agent.scenario.scenario_policy import decide
from app.agent.scenario.scenario_router import LABELS, RECORD_QUERY_GUIDE, Scenario, build_ask
from app.core import get_settings

A, E = Scenario.WORK_REGISTER, Scenario.AUDIT_QUERY
ENTRY = (A, E)  # 채팅 범위 기본값 (F는 구현하지 않는다)
PERIOD = "2026-10-01~2026-10-31"

ROUTE, GUIDE, WORK, ASK = "보고서 분기(E)", "기록 질문 안내", "작업 등록(A)", "되묻기"


def _kind(d) -> str:
    if d.scenario is E:
        return ROUTE
    if d.scenario is A:
        return WORK
    assert d.scenario is Scenario.UNKNOWN
    return GUIDE if d.redirect is E else ASK


# (문장, 바뀐 뒤, 스위치를 끄면(변경 전))
TABLE = [
    ("감사 보고서 만들어줘", ROUTE, ROUTE),  # 1
    ("지난달 감사 대응 보고서", ROUTE, ROUTE),  # 2
    ("전월 감사 대응", ROUTE, ROUTE),  # 3 (기간 '전월'은 기간 읽기가 못 읽어 되묻는다)
    ("지난달 감사 자료 작성해줘", ROUTE, ROUTE),  # 4
    ("10월 감사 보고서", ROUTE, ROUTE),  # 5
    ("지난달 감사 대응용 실시율 통계 보여줘", GUIDE, ROUTE),  # 6
    ("지난달 실시율 알려줘", GUIDE, ROUTE),  # 7
    ("지난달 측정값 빠진 허가서 알려줘", GUIDE, ROUTE),  # 8
    ("지난달 TBM을 안 한 작업은?", GUIDE, ROUTE),  # 9
    ("이번 주 승인 대기 허가서 조회", GUIDE, ROUTE),  # 10
    ("아차사고 보고서 작성해줘", WORK, ROUTE),  # 11
    ("TBM 결과 보고서 정리해줘", WORK, ROUTE),  # 12
    ("감사 보고서 등록해줘", ASK, ASK),  # 13
    ("내일 용접 작업 등록해줘 감사합니다", WORK, WORK),  # 14
    ("내일 용접 작업 등록해줘, 필요한 서류 알려줘", WORK, WORK),  # 15 (등록 문장의 '알려줘'는 A 그대로)
    ("감사합니다 지난달 실시율 알려줘", GUIDE, ROUTE),  # 16
    ("지난달 감사 대응 보고서 만들어줘 감사합니다", ROUTE, ROUTE),  # 17
    ("작업 완료 보고서 작성해줘", WORK, ROUTE),  # 18
    ("감사 대응", ROUTE, ROUTE),  # 19
]


@pytest.mark.parametrize("text,after,_before", TABLE, ids=[t[0] for t in TABLE])
def test_sentence_table_after(text, after, _before):
    assert _kind(decide(text, mode="VERIFY", allowed=ENTRY)) == after


@pytest.mark.parametrize("text,_after,before", TABLE, ids=[t[0] for t in TABLE])
def test_switch_off_restores_the_old_rule(monkeypatch, text, _after, before):
    monkeypatch.setattr(scenario_policy, "AUDIT_QUERY_SPLIT", False)
    assert _kind(decide(text, mode="VERIFY", allowed=ENTRY)) == before


def test_audit_priority_off_also_turns_the_split_off(monkeypatch):
    # AUDIT_PRIORITY=False는 11/19 이전 규칙이다. 분리 규칙은 그 위에서만 동작한다
    monkeypatch.setattr(scenario_policy, "AUDIT_PRIORITY", False)
    d = decide("지난달 실시율 알려줘", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is E and d.redirect is None


# ── 안전 조건: 작업 등록 문장이 새지 않는다 ──


@pytest.mark.parametrize(
    "text",
    [
        "내일 3층 용접 작업 필요한 허가 알려줘",  # E 낱말이 없다 → 이전처럼 A
        "용접 작업 하려는데 안전모 안 한 사람이 있어요",  # '안 한'이 있어도 E 낱말이 없다 → A
        "오늘 13시 작업 알려줘",
    ],
)
def test_work_sentences_without_e_words_stay_a(text):
    assert decide(text, mode="VERIFY", allowed=ENTRY).scenario is A


def test_register_or_clock_blocks_the_record_guide():
    # E 낱말(어제) + 기록 질문말(알려) + A의 강한 단서(등록·9시) → 안내가 아니라 이전처럼 되묻는다
    d = decide("어제 하던 용접 이어서 내일 오전 9시에 등록, 알려줘", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.redirect is None and d.ask


def test_report_word_beats_record_word():
    # 보고서를 만들어 달라는 말이 함께 있으면 기록 질문이 아니다
    assert decide("감사 대응 실시율 보고서 만들어줘", mode="VERIFY", allowed=ENTRY).scenario is E


def test_report_word_without_audit_is_not_a_record_question():
    # 기록 질문말·E 낱말이 있어도 보고서 요청말이 같이 있으면 안내가 아니다 ('감사'가 없으니 보고서 분기도 아니다 → 되묻는다)
    d = decide("지난달 실시율 통계 보고서 만들어줘", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.redirect is None and d.ask != RECORD_QUERY_GUIDE


@pytest.mark.parametrize("text", ["지난달 실시율 통계 자료 알려줘", "지난달 실시율 보고서 알려줘"])
def test_record_words_with_a_report_noun_are_guided(text):
    # v2(AUDIT_QUERY_STRICT): 보고서·자료(명사)만 있고 만들어 달라는 동사가 없으면 기록 질문말이 우선한다 (v1에서는 E)
    assert _kind(decide(text, mode="VERIFY", allowed=ENTRY)) == GUIDE


@pytest.mark.parametrize("text", ["오늘 9시 감사 대응", "내일 9시 감사 대응"])
def test_audit_response_with_clock_still_goes_to_e(text):
    # '감사' + '대응'이면 시각 단서(N시)가 있어도 E (11/19 규칙 그대로). 등록말이 있으면 아니다
    assert decide(text, mode="VERIFY", allowed=ENTRY).scenario is E
    assert decide(text + " 등록", mode="VERIFY", allowed=ENTRY).scenario is not E


def test_report_alone_is_not_audit():
    # '감사' 없이 '보고서'만 있으면 감사가 아니다 (이전에는 E)
    assert decide("보고서", mode="VERIFY", allowed=ENTRY).scenario is Scenario.UNKNOWN
    assert decide("위험성평가 보고서 만들어줘", mode="VERIFY", allowed=ENTRY).scenario is A


def test_audit_report_list_request_is_guided():
    # v2: '감사 보고서 목록 보여줘'는 기록 질문이다 (v1에서는 보고서 분기였다)
    assert _kind(decide("감사 보고서 목록 보여줘", mode="VERIFY", allowed=ENTRY)) == GUIDE


# ── 안내 결정의 모양 ──


def test_record_guide_decision_shape():
    d = decide("지난달 실시율 알려줘", mode="VERIFY", allowed=ENTRY)
    assert d.scenario is Scenario.UNKNOWN and d.redirect is E and d.candidate is None
    assert d.source == "fallback" and d.ask == RECORD_QUERY_GUIDE
    assert "기록 질문" in d.reason


def test_record_guide_without_chat_scope_limit_and_in_guide_mode():
    assert _kind(decide("지난달 실시율 알려줘", mode="VERIFY", allowed=None)) == GUIDE
    # F가 채팅 범위에 없으면 GUIDE 현장도 말로 가른다 (guide_off)
    assert _kind(decide("지난달 실시율 알려줘", mode="GUIDE", allowed=ENTRY)) == GUIDE


def test_record_guide_not_used_when_e_is_out_of_chat_scope():
    # E가 채팅 범위 밖이면 "감사 대응 보고서를 만들 수 있어요"가 사실이 아니다 → 이전처럼 범위 밖 안내
    d = decide("지난달 실시율 알려줘", mode="VERIFY", allowed=(A,))
    assert d.scenario is Scenario.UNKNOWN and d.redirect is E
    assert d.ask != RECORD_QUERY_GUIDE and "전용 화면" in d.ask


# ── 안내 문구 ──


def test_record_query_guide_text():
    assert "기록 질문" in RECORD_QUERY_GUIDE and "작업 등록" in RECORD_QUERY_GUIDE and "감사 대응 보고서" in RECORD_QUERY_GUIDE
    assert "전용 화면" not in RECORD_QUERY_GUIDE and "\n- " not in RECORD_QUERY_GUIDE  # 어딘지 말하고, 선택지를 되풀이하지 않는다
    assert not any(code in RECORD_QUERY_GUIDE for code in ("AUDIT_QUERY", "AUDIT_REPORT", "route"))  # 내부 코드를 보이지 않는다


def test_labels_do_not_repeat_the_record_item():
    assert LABELS[E] == "감사 대응 보고서 만들기"
    assert "기록 조회" not in LABELS[E]
    ask = build_ask(None, allowed=ENTRY)
    assert LABELS[A] in ask and LABELS[E] in ask and "기록 조회" not in ask


def test_label_texts_classify_as_intended():
    # 선택지를 눌러 문구가 그대로 전송돼도 알맞게 갈린다
    assert decide(LABELS[E], mode="VERIFY", allowed=ENTRY).scenario is E
    assert decide(LABELS[A], mode="VERIFY", allowed=ENTRY).scenario is A


def test_route_off_redirect_text_is_unchanged_except_the_label():
    # E가 채팅 범위 밖일 때의 이전 안내 문구 틀은 그대로다 (이름만 새 선택지 이름)
    assert build_ask(redirect=E, allowed=(A,)).startswith(f"'{LABELS[E]}'은(는) 채팅이 아니라 전용 화면에서 이용해 주세요.")


# ── work-register 응답 ──


def _client(monkeypatch, tmp_path, *, route: bool = True):
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
    get_settings.cache_clear()
    from app.main import create_app

    return TestClient(create_app())


def _post(client, message, slots=None):
    body = {"runId": 1, "siteId": 1, "mode": "VERIFY", "message": message, "slots": slots}
    response = client.post("/api/v1/agent/work-register", json=body)
    assert response.status_code == 200, response.text
    return response.json()["data"]


@pytest.mark.parametrize("text", ["지난달 실시율 알려줘", f"{PERIOD} 승인 대기 허가서 조회", "지난달 TBM을 안 한 작업은?"])
def test_record_question_gets_the_guide_and_no_report_route(monkeypatch, tmp_path, text):
    data = _post(_client(monkeypatch, tmp_path), text)
    assert data["runStatus"] == "WAITING_INPUT"
    assert data["route"] is None and data["drafts"] is None and data["steps"] == []
    assert data["ask"]["message"] == RECORD_QUERY_GUIDE and data["ask"]["missingFields"] == []
    assert data["slots"]["pendingText"] is None  # 다른 화면의 일이다. 다음 말은 새로 분류한다


def test_report_request_still_routes(monkeypatch, tmp_path):
    data = _post(_client(monkeypatch, tmp_path), f"{PERIOD} 감사 보고서 만들어줘")
    assert data["runStatus"] == "DONE" and data["route"]["target"] == "AUDIT_REPORT"
    assert (data["route"]["dateFrom"], data["route"]["dateTo"]) == ("2026-10-01", "2026-10-31")


def test_guide_then_report_request_in_the_next_turn(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    first = _post(client, "지난달 실시율 알려줘")
    # 안내 뒤의 다음 말은 앞선 말과 이어 붙이지 않고 새로 분류한다 (안내는 되묻기가 아니다)
    second = _post(client, f"{PERIOD} 감사 보고서 만들어줘", slots=first["slots"])
    assert second["runStatus"] == "DONE" and second["route"]["dateFrom"] == "2026-10-01"
