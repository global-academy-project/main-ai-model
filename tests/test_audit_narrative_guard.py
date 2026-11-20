"""감사 보고서 항목 문장(LLM 후보) 규칙 가드: 평가어·날짜·핵심 값·교체 범위·호출 대상·시간 예산.

모델은 부르지 않는다: 가짜 생성기를 넣는다. 가드를 끄면(NARRATIVE_GUARD=False) 이전 방식과 같아야 한다.
"""

import copy
import re

import pytest

from app.agent import audit_narrative_guard as guard
from app.agent.audit_compose import compose_audit_report
from app.schemas.audit_compose import AuditComposeInput, ComposeSentence

REQUEST = {'reportId': 12,
 'siteId': 3,
 'dateFrom': '2026-10-01',
 'dateTo': '2026-10-31',
 'items': [{'queryType': 'risk_assessment_rate',
            'title': '위험성평가 실시율',
            'stats': {'works': 40,
                      'written': 38,
                      'approved': 36,
                      'missing': 2,
                      'rate': 95.0,
                      'approvedRate': 90.0},
            'total': 1,
            'truncated': False,
            'records': [],
            'evidence': []},
           {'queryType': 'permit_approval_statistics',
            'title': '작업허가 승인 현황',
            'stats': {'permits': 25,
                      'requested': 25,
                      'approved': 21,
                      'rejected': 2,
                      'canceled': 1,
                      'blocked': 1,
                      'approvalRate': 84.0}},
           {'queryType': 'corrective_action_rate',
            'title': '지적사항 조치율',
            'stats': {'actions': 10,
                      'completed': 6,
                      'verified': 4,
                      'open': 4,
                      'overdue': 1,
                      'completionRate': 60.0},
            'records': [{'id': 77, 'sourceType': 'INSPECTION', 'sourceId': 5}]},
           {'queryType': 'blocked_records',
            'title': '차단 기록',
            'stats': {'count': 1},
            'records': [{'permitId': 901, 'evaluationId': 44}],
            'evidence': [{'type': 'PERMIT', 'id': 901}]},
           {'queryType': 'missing_measurements',
            'title': '측정값 없이 승인된 허가',
            'stats': {'count': 0},
            'records': []}]}


def compose(generator=None, request=None):
    data = AuditComposeInput.model_validate(copy.deepcopy(request or REQUEST))
    return compose_audit_report(data, narrative_generator=generator)


def items(result):
    return {item.query_type: item for group in result.groups for item in group.items}


def codes(result):
    return [v.code for v in result.verification]


RISK_OK = ["해당 기간 작업 40건 중 위험성평가를 작성한 작업은 38건(95.0%), 승인된 작업은 36건(90.0%)이고, 기록이 없는 작업은 2건임."]


def risk_generator(candidates):
    calls = []

    def generate(item):
        calls.append(item.query_type)
        return list(candidates) if item.query_type == "risk_assessment_rate" else None

    generate.calls = calls
    return generate


# ── 호출 대상과 채택 ──────────────────────────────────────────


def test_only_review_items_are_asked():
    calls = []

    def generate(item):
        calls.append(item.query_type)
        return ["(아무 문장)"]

    compose(generate)
    assert calls == ["risk_assessment_rate", "corrective_action_rate"]


def test_valid_candidate_is_adopted_and_basis_sentence_stays_first():
    base = items(compose())["risk_assessment_rate"]
    result = compose(risk_generator(RISK_OK))
    item = items(result)["risk_assessment_rate"]
    assert item.sentences[0].text == base.sentences[0].text  # 기준(법적 근거) 문장은 템플릿 그대로
    assert item.sentences[0].source == "TEMPLATE"
    assert [s.source for s in item.sentences[1:]] == ["LLM"]
    assert item.sentences[1].text == RISK_OK[0]
    assert "NARRATIVE_REPLACED" not in codes(result)
    # 이 시험의 생성기는 다른 REVIEW 항목에 None을 돌려줘 그 항목만 FALLBACK이 된다 (위험성평가 항목은 아님)
    assert all("위험성평가" not in v.message for v in result.verification if v.code == "NARRATIVE_FALLBACK")


def test_other_items_keep_template_sentences():
    base = items(compose())
    result = items(compose(risk_generator(RISK_OK)))
    for query_type, item in result.items():
        if query_type != "risk_assessment_rate":
            assert [s.model_dump() for s in item.sentences] == [s.model_dump() for s in base[query_type].sentences]


# ── 거르는 것 ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "작업 40건 중 38건(95.0%) 36건(90.0%) 2건이며 안전 규정을 위반한 사례가 있음.",  # 평가어
        "작업 40건 중 38건(95.0%) 36건(90.0%) 2건이며 부적정함.",  # 평가어
        "작업 40건 중 38건(95.0%) 36건(90.0%) 2건이며 2025-03-15에 점검함.",  # 지어낸 날짜
        "작업 40건 중 38건(95.0%) 36건(90.0%) 2건이며 10월 3일에 점검함.",  # 한글 날짜 표현
        "작업 40건 중 38건(95.0%) 36건(90.0%) 2건이며 99건을 확인함.",  # 항목에 없는 숫자
        "작업 40건 중 38건(95.0%)임.",  # 핵심 값(36·90.0·2)이 빠짐
        "",  # 빈 문장
        "가" * 201,  # 너무 긴 문장
    ],
)
def test_bad_candidate_keeps_template(text):
    base = items(compose())["risk_assessment_rate"]
    result = compose(risk_generator([text]))
    item = items(result)["risk_assessment_rate"]
    assert [s.model_dump() for s in item.sentences] == [s.model_dump() for s in base.sentences]
    assert "NARRATIVE_REPLACED" in codes(result)


def test_five_or_more_sentences_are_rejected():
    result = compose(risk_generator([RISK_OK[0]] * 5))
    assert "NARRATIVE_REPLACED" in codes(result)
    assert all(s.source == "TEMPLATE" for s in items(result)["risk_assessment_rate"].sentences)


def test_verification_message_never_contains_candidate_text():
    result = compose(risk_generator(["안전 규정을 위반한 사례가 있음. 2025-03-15"]))
    for v in result.verification:
        assert "위반" not in v.message and "2025-03-15" not in v.message


def test_request_period_dates_are_allowed():
    text = RISK_OK[0] + " 점검 기간은 2026-10-01부터 2026-10-31까지임."
    # 날짜는 숫자 검사에서 빠지고, 입력 기간의 날짜라 허용된다
    result = compose(risk_generator([text]))
    assert "NARRATIVE_REPLACED" not in codes(result)
    assert items(result)["risk_assessment_rate"].sentences[1].source == "LLM"


def test_template_sentences_themselves_have_no_banned_words():
    for item in items(compose()).values():
        for sentence in item.sentences:
            assert guard.banned_words_in(sentence.text) == [], sentence.text


# ── 교체 범위 ─────────────────────────────────────────────────


def test_split_keeps_basis_evidence_truncated_and_formula_sentences():
    from app.schemas.audit_compose import EvidenceRef

    sentences = [
        ComposeSentence(text="기준 문장."),
        ComposeSentence(text="현황 문장 3건임."),
        ComposeSentence(text="근거 있는 문장 1건임.", evidence=[EvidenceRef(type="PERMIT", id=9)]),
        ComposeSentence(text="전체 120건 중 100건만 표시함."),
        ComposeSentence(text="승인률(승인÷(승인+반려))은 91.3%임."),
    ]
    head, replaceable, kept = guard.split_sentences(sentences)
    assert [s.text for s in head] == ["기준 문장."]
    assert [s.text for s in replaceable] == ["현황 문장 3건임."]
    assert [s.text for s in kept] == [sentences[2].text, sentences[3].text, sentences[4].text]


def test_evidence_sentence_survives_adoption(monkeypatch):
    monkeypatch.setattr(guard, "NARRATIVE_STATES", ("REVIEW", "REFERENCE"))
    base = items(compose())["blocked_records"]
    assert any(s.evidence for s in base.sentences)  # 이 요청의 차단 기록 문장에는 근거가 붙어 있다

    def generate(item):
        return ["차단은 허가 단계에서 규칙에 따라 승인을 막은 기록임."] if item.query_type == "blocked_records" else None

    result = compose(generate)
    item = items(result)["blocked_records"]
    assert any(s.evidence for s in item.sentences)  # 근거 문장은 그대로
    assert item.sentences[0].text == base.sentences[0].text
    assert any(s.source == "LLM" for s in item.sentences)


def test_item_without_replaceable_sentences_is_not_asked(monkeypatch):
    monkeypatch.setattr(guard, "NARRATIVE_STATES", ("OK",))
    calls = []

    def generate(item):
        calls.append(item.query_type)
        return ["문장"]

    compose(generate)
    # missing_measurements(OK)는 현황 문장 1개가 있어 호출되고, 그 밖 상태 항목은 호출되지 않는다
    assert calls == ["missing_measurements"]


# ── 실패·시간 ─────────────────────────────────────────────────


def test_generator_error_falls_back_and_continues():
    calls = []

    def generate(item):
        calls.append(item.query_type)
        if item.query_type == "risk_assessment_rate":
            raise RuntimeError("모델 오류")
        return ["작업 중 미완료 개선조치는 4건이며 기한 경과는 1건, 개선조치 10건 중 완료 6건(60.0%), 검증 완료 4건임."]

    result = compose(generate)
    assert calls == ["risk_assessment_rate", "corrective_action_rate"]
    assert "NARRATIVE_FALLBACK" in codes(result)
    assert items(result)["risk_assessment_rate"].state == "REVIEW"


def test_non_list_return_falls_back():
    result = compose(lambda item: "문자열")
    assert "NARRATIVE_FALLBACK" in codes(result)


def test_time_budget_stops_remaining_items(monkeypatch):
    monkeypatch.setattr(guard, "NARRATIVE_BUDGET_SECONDS", -1.0)
    calls = []

    def generate(item):
        calls.append(item.query_type)
        return ["문장"]

    result = compose(generate)
    assert calls == []
    assert "NARRATIVE_FALLBACK" in codes(result)
    assert items(result)["risk_assessment_rate"].sentences[0].source == "TEMPLATE"


def test_no_generator_means_no_change():
    a = compose()
    b = compose(None)
    assert a.model_dump(exclude={"meta"}) == b.model_dump(exclude={"meta"})


# ── 끄기 ──────────────────────────────────────────────────────


def test_guard_off_restores_previous_behaviour(monkeypatch):
    monkeypatch.setattr(guard, "NARRATIVE_GUARD", False)
    calls = []

    def generate(item):
        calls.append(item.query_type)
        return ["안전 규정을 위반한 사례가 있음."]

    result = compose(generate)
    # 이전 방식: 상태와 무관하게 UNAVAILABLE이 아닌 5개 항목에 모두 묻고, 평가어도 막지 않으며, 문장을 통째로 교체한다
    assert len(calls) == 5
    assert [s.text for s in items(result)["risk_assessment_rate"].sentences] == ["안전 규정을 위반한 사례가 있음."]
