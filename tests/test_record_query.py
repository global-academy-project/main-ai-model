"""기록질문 계획, Spring 호출, 검증된 답변 계약."""

import pytest

from app.agent.record_query import RecordQueryAgent
from app.exceptions import AgentRunError
from app.llm.llm_fake import FakeLLMClient, text_response
from app.schemas.record_query import RecordQueryRequest


def request(**overrides) -> RecordQueryRequest:
    values = {
        "runId": 51,
        "siteId": 1,
        "message": "지난달 승인된 화기작업허가 중 가스측정값 누락 기록을 보여줘",
        "context": None,
        "mode": "VERIFY",
    }
    values.update(overrides)
    return RecordQueryRequest.model_validate(values)


def plan(query_type="missing_measurements", params=None):
    params = params or {"from": "2026-10-01", "to": "2026-10-31", "permitType": "HOT"}
    return text_response(f'{{"queryType":"{query_type}","params":{__import__("json").dumps(params)}}}')


def result(total=1):
    return {
        "queryType": "missing_measurements",
        "conditions": {
            "siteId": 1,
            "from": "2026-10-01",
            "to": "2026-10-31",
            "permitType": "HOT",
        },
        "total": total,
        "truncated": False,
        "summary": {"count": total},
        "records": [{"id": 5001, "permitNo": "HOT-20261014-002"}] if total else [],
        "evidence": [{"type": "PERMIT", "id": 5001}] if total else [],
    }


def test_answers_only_from_search_result_and_injects_run_scope():
    calls = []

    def search(arguments):
        calls.append(dict(arguments))
        return result()

    agent = RecordQueryAgent(llm=FakeLLMClient([plan()]), search_records=search)
    response = agent.run(request())

    assert response.run_status == "DONE"
    assert response.answer == "조회 조건에 맞는 가스측정값 누락 기록은 총 1건입니다."
    assert response.evidence[0].model_dump() == {
        "type": "PERMIT",
        "id": 5001,
        "label": "HOT-20261014-002",
    }
    assert calls == [{
        "run_id": 51,
        "site_id": 1,
        "query_type": "missing_measurements",
        "params": {"from": "2026-10-01", "to": "2026-10-31", "permitType": "HOT"},
    }]
    assert [step.name for step in response.steps] == ["query.plan", "search_records", "verification"]


def test_zero_records_does_not_invent_evidence():
    agent = RecordQueryAgent(llm=FakeLLMClient([plan()]), search_records=lambda _: result(0))
    response = agent.run(request())
    assert response.answer == "조회 조건에 맞는 가스측정값 누락 기록은 없습니다."
    assert response.evidence == []


@pytest.mark.parametrize("params", [
    {"from": "2026-10-01"},
    {"from": "2026-10-31", "to": "2026-10-01"},
    {"from": "2025-01-01", "to": "2026-10-01"},
    {"from": "2026-10-01", "to": "2026-10-31", "unknown": "x"},
])
def test_invalid_or_incomplete_plan_asks_for_clarification(params):
    called = False

    def search(_):
        nonlocal called
        called = True
        return result()

    response = RecordQueryAgent(
        llm=FakeLLMClient([plan(params=params)]), search_records=search
    ).run(request())
    assert response.run_status == "WAITING_INPUT"
    assert response.ask is not None
    assert not called


def test_rejects_search_result_from_another_site():
    bad = result()
    bad["conditions"]["siteId"] = 2
    agent = RecordQueryAgent(llm=FakeLLMClient([plan()]), search_records=lambda _: bad)
    with pytest.raises(AgentRunError) as error:
        agent.run(request())
    assert error.value.cause.code == "TOOL_OUTPUT_INVALID"
    assert error.value.steps[-1].name == "search_records"


def test_follow_up_context_is_given_to_planner():
    llm = FakeLLMClient([plan()])
    agent = RecordQueryAgent(llm=llm, search_records=lambda _: result())
    agent.run(request(context={"lastQueryType": "missing_tbm", "lastParams": {"from": "2026-10-01"}}))
    prompt = llm.calls[0]["messages"][0].content
    assert "lastQueryType" in prompt
    assert "missing_tbm" in prompt
