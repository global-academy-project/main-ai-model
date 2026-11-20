"""기록 질문 계획 규칙: 분명한 질문만 코드로 정하고, 애매하면 None(= D의 LLM 계획). 모델·네트워크는 쓰지 않는다."""

from datetime import date

import pytest

from app.agent.record_plan_rules import match_types, rule_plan
from app.agent.record_query import RecordQueryAgent
from app.agent.record_query_rules_agent import RuleFirstRecordQueryAgent
from app.llm.llm_fake import FakeLLMClient, text_response
from app.schemas.record_query import RecordQueryRequest

TODAY = date(2026, 11, 20)  # 금요일


def plan(message, context=None):
    return rule_plan(message, context, TODAY)


@pytest.mark.parametrize(
    ("message", "query_type", "params"),
    [
        ("지난달 가스측정값이 빠진 화기허가", "missing_measurements", {"from": "2026-10-01", "to": "2026-10-31", "permitType": "HOT"}),
        ("지난달 승인된 화기작업허가 중 가스측정값 누락 기록을 보여줘", "missing_measurements", {"from": "2026-10-01", "to": "2026-10-31", "permitType": "HOT"}),
        ("10월 위험성평가 실시 현황", "risk_assessment_rate", {"from": "2026-10-01", "to": "2026-10-31"}),
        ("이번 주 TBM 안 한 작업", "missing_tbm", {"from": "2026-11-16", "to": "2026-11-20"}),
        ("지난달 고소작업 TBM 누락", "missing_tbm", {"from": "2026-10-01", "to": "2026-10-31", "workType": "HEIGHT"}),
        ("지난달 차단된 허가 몇 건이야", "blocked_records", {"from": "2026-10-01", "to": "2026-10-31"}),
        ("지난달 작업허가 승인 현황", "permit_approval_statistics", {"from": "2026-10-01", "to": "2026-10-31"}),
        ("지난달 밀폐공간 허가 승인 통계", "permit_approval_statistics", {"from": "2026-10-01", "to": "2026-10-31", "permitType": "CONFINED"}),
        ("지난달 시정조치 미완료 건", "open_corrective_actions", {"from": "2026-10-01", "to": "2026-10-31"}),
        ("지난주 반려 후 수정된 기록", "rejected_revisions", {"from": "2026-11-09", "to": "2026-11-15"}),
        ("지난달 부적합 점검 기록", "failed_inspections", {"from": "2026-10-01", "to": "2026-10-31"}),
        ("2026-10-01부터 2026-10-15까지 TBM 누락", "missing_tbm", {"from": "2026-10-01", "to": "2026-10-15"}),
    ],
)
def test_clear_questions_are_planned_by_rules(message, query_type, params):
    result = plan(message)
    assert result is not None, message
    assert (result.query_type, result.params) == (query_type, params)


@pytest.mark.parametrize(
    "message",
    [
        "TBM 안 한 작업",  # 기간 없음
        "최근 3개월 위험성평가 실시 현황",  # 읽기기가 안 읽는 기간 표현
        "지난달과 이번 달 TBM 누락",  # 기간 둘
        "지난달 TBM 누락 허가 승인 현황",  # 유형이 둘
        "지난달 승인 대기 허가",  # 값을 모르는 조건(대상 종류)
        "지난달 승인자 통계",  # 규칙 대상 유형이 아님
        "지난달 기록 변경 이력",  # 기록 번호가 필요
        "지난달 화기 밀폐 가스측정 누락",  # 종류가 둘
        "2025년 1월 1일부터 2026년 11월 1일까지 TBM 누락",  # 365일 초과
        "지난달 3구역 부적합 점검",  # 구역 조건
        "지난달 기한 지난 시정조치 미완료",  # 값을 모르는 조건(상태)
        "그중 화기만 보여줘",  # 이어 묻기
        "오늘 날씨 알려줘",
        "안녕하세요",
    ],
)
def test_unclear_questions_fall_through(message):
    assert plan(message) is None, message


def test_match_types_is_conservative():
    # 제외어로 빠진 유형(허가 승인 현황)이 같이 보이면 규칙 계획을 포기한다
    assert match_types("지난달 가스측정 누락 허가 승인 현황") == ["missing_measurements"]
    assert plan("지난달 가스측정 누락 허가 승인 현황") is None
    assert plan("지난달 승인자별 허가 승인 통계") is None
    assert match_types("그냥 허가") == []


# ── 에이전트: 규칙이 정하면 모델을 안 부르고, 못 정하면 D의 계획으로 넘긴다 ──


def request(message, **overrides):
    values = {"runId": 51, "siteId": 1, "message": message, "context": None, "mode": "VERIFY"}
    values.update(overrides)
    return RecordQueryRequest.model_validate(values)


def result(query_type="risk_assessment_rate", total=2):
    return {
        "queryType": query_type,
        "conditions": {"siteId": 1},
        "total": total,
        "truncated": False,
        "summary": {"count": total},
        "records": [],
        "evidence": [],
    }


def test_agent_rule_path_and_fallback_path():
    calls = []

    def search(arguments):
        calls.append(dict(arguments))
        return result(arguments["query_type"])

    llm = FakeLLMClient([text_response('{"decision":"CLARIFY","reason":"조회할 기간을 알려 주세요."}')])
    agent = RuleFirstRecordQueryAgent(llm=llm, search_records=search)

    # 기간 표현이 없으면 규칙이 못 정하고 D의 모델 계획으로 넘어간다 (되묻기도 거기서)
    response = agent.run(request("TBM을 안 한 작업을 알려줘"))
    assert response.run_status == "WAITING_INPUT"
    assert "조회할 기간" in response.ask.message
    assert len(llm.calls) == 1
    assert calls == []


def test_agent_rule_path_does_not_call_model_and_injects_run_scope():
    calls = []

    def search(arguments):
        calls.append(dict(arguments))
        return result(arguments["query_type"])

    llm = FakeLLMClient([])
    agent = RuleFirstRecordQueryAgent(llm=llm, search_records=search)
    response = agent.run(request("2026-10-01부터 2026-10-31까지 위험성평가 실시 현황"))
    assert response.run_status == "DONE"
    assert llm.calls == []
    assert len(calls) == 1
    assert calls[0]["run_id"] == 51 and calls[0]["site_id"] == 1  # 실행 요청 값만 쓴다
    assert calls[0]["query_type"] == "risk_assessment_rate"
    assert set(calls[0]["params"]) == {"from", "to"}
    assert [step.name for step in response.steps] == ["query.plan", "search_records", "verification"]


def test_agent_out_of_scope_is_left_to_d_without_model():
    llm = FakeLLMClient([])
    agent = RuleFirstRecordQueryAgent(llm=llm, search_records=lambda _: result())
    response = agent.run(request("오늘 날씨 알려줘"))
    assert response.run_status == "WAITING_INPUT"
    assert llm.calls == []


def test_agent_subclass_still_matches_d_internals():
    # D가 이름을 바꾸면 여기서 알린다 (deps가 _llm·_search_records를 읽는다)
    agent = RecordQueryAgent(llm=FakeLLMClient([]), search_records=lambda _: result())
    assert hasattr(agent, "_llm") and hasattr(agent, "_search_records")
    assert hasattr(RecordQueryAgent, "_valid_params") and hasattr(RecordQueryAgent, "_is_obviously_out_of_scope")
