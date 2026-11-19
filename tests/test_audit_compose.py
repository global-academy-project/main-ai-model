"""감사 대응 보고서 구성기 시험 (LLM·DB·Spring 없이 mock 입력으로)."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agent import audit_compose, audit_compose_criteria as criteria
from app.agent.audit_compose import compose_audit_report
from app.agent.audit_compose_text import SPECS, action_status_ko, date_part
from app.agent.audit_compose_check import (
    check_corrective,
    check_permit,
    check_risk,
    numbers_in,
    percent,
    rate_differs,
    unexpected_numbers,
)
from app.api.v1.agent.audit_compose import router
from app.schemas.audit_compose import AuditComposeInput
from scripts.preview_audit_compose import main as preview_main
from scripts.preview_audit_compose import render_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "audit_compose"
NOW = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone(timedelta(hours=9)))


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def compose(data: dict):
    return compose_audit_report(AuditComposeInput.model_validate(data), now=NOW)


def items_by_type(result) -> dict:
    return {i.query_type: i for g in result.groups for i in g.items}


def codes(result) -> list[str]:
    return [v.code for v in result.verification]


def all_text(result) -> str:
    return json.dumps(result.model_dump(by_alias=True), ensure_ascii=False)


# ---------------------------------------------------------------- 숫자 도구


@pytest.mark.parametrize(
    "part,whole,expected",
    [(2, 3, 66.7), (1, 16, 6.3), (1, 8, 12.5), (5, 5, 100.0), (0, 7, 0.0), (1, 3, 33.3), (7, 12, 58.3)],
)
def test_percent_rounds_half_up(part, whole, expected):
    assert percent(part, whole) == expected


def test_percent_zero_denominator_is_none():
    assert percent(0, 0) is None
    assert percent(3, 0) is None


def test_numbers_in_ignores_dates():
    assert numbers_in("2026-10-01부터 2026-10-31까지 12건(83.3%)") == ["12", "83.3"]
    assert numbers_in("제36조") == ["36"]
    assert unexpected_numbers("작업 12건 중 3건", {"12", "3"}) == []
    assert unexpected_numbers("작업 12건 중 4건", {"12", "3"}) == ["4"]


def test_rate_differs_cases():
    assert rate_differs(83.3, 83.3) is False
    assert rate_differs(83.34, 83.3) is False
    assert rate_differs(83.0, 83.3) is True
    assert rate_differs(None, None) is False
    assert rate_differs(None, 50.0) is True
    assert rate_differs(50.0, None) is True
    assert rate_differs("83.3", 83.3) is True


def test_stats_checks_reject_bad_values():
    for bad in ({"works": True, "written": 1, "approved": 1, "missing": 0},
                {"works": -1, "written": 0, "approved": 0, "missing": 0},
                {"works": 3.0, "written": 1, "approved": 1, "missing": 2}):
        with pytest.raises(Exception) as exc:
            check_risk(bad)
        assert getattr(exc.value, "code") == "STATS_INVALID"
    with pytest.raises(Exception) as exc:
        check_risk({"works": 3, "written": 2, "approved": 1, "missing": 0})
    assert exc.value.code == "STATS_INCONSISTENT"
    with pytest.raises(Exception):
        check_permit({"permits": 3, "requested": 2, "approved": 2, "rejected": 1, "canceled": 0, "blocked": 0})
    with pytest.raises(Exception):
        check_corrective({"actions": 3, "completed": 1, "verified": 2, "open": 2, "overdue": 0})
    with pytest.raises(Exception):
        check_corrective({"actions": 3, "completed": 1, "verified": 1, "open": 1, "overdue": 0})
    with pytest.raises(Exception):
        check_corrective({"actions": 3, "completed": 1, "verified": 1, "open": 2, "overdue": 3})


# ---------------------------------------------------------------- 상태 기준 (임시, 한 곳)


def test_state_criteria_boundaries():
    assert criteria.risk_state(10, 0) == "OK"
    assert criteria.risk_state(10, 1) == "REVIEW"
    assert criteria.permit_state(0) == "OK"
    assert criteria.permit_state(1) == "REFERENCE"
    assert criteria.missing_measurement_state(0) == "OK"
    assert criteria.missing_measurement_state(1) == "REVIEW"
    assert criteria.blocked_state(0) == "OK"
    assert criteria.blocked_state(1) == "REFERENCE"
    assert criteria.corrective_state(0, 0, 0) == "OK"
    assert criteria.corrective_state(5, 0, 0) == "OK"
    assert criteria.corrective_state(5, 2, 0) == "REFERENCE"
    assert criteria.corrective_state(5, 2, 1) == "REVIEW"
    assert set(criteria.STATE_LABELS) == {"REVIEW", "REFERENCE", "OK", "UNAVAILABLE"}


# ---------------------------------------------------------------- 정상 입력


def test_normal_structure_and_order():
    r = compose(load("normal"))
    assert r.form_type == "AUDIT_REPORT" and r.schema_version == "v1"
    assert r.status == "COMPLETE"
    assert r.meta.generated_at == "2026-11-19T15:00:00+09:00"
    assert [g.seq for g in r.groups] == [1, 2, 3]
    assert [g.no for g in r.groups] == ["AR-1-202610-01", "AR-1-202610-02", "AR-1-202610-03"]
    assert [[i.query_type for i in g.items] for g in r.groups] == [
        ["risk_assessment_rate", "permit_approval_statistics"],
        ["near_miss_receipt", "near_miss_action_link"],
        ["missing_measurements", "blocked_records", "corrective_action_rate"],
    ]
    assert [i.seq for g in r.groups for i in g.items] == [1, 2, 3, 4, 5, 6, 7]
    for g in r.groups:
        assert g.responder_opinion is None and g.requested_action is None


def test_normal_states_and_summary_rows():
    r = compose(load("normal"))
    by = items_by_type(r)
    assert {k: v.state for k, v in by.items()} == {
        "risk_assessment_rate": "REVIEW",
        "permit_approval_statistics": "REFERENCE",
        "near_miss_receipt": "UNAVAILABLE",
        "near_miss_action_link": "UNAVAILABLE",
        "missing_measurements": "REVIEW",
        "blocked_records": "REFERENCE",
        "corrective_action_rate": "REVIEW",
    }
    assert by["risk_assessment_rate"].state_label == "확인 필요"
    rows = {row.area: row for g in r.summary.groups for row in g.rows}
    risk = rows["위험성평가 실시"]
    assert (risk.checked, risk.count, risk.rate_pct) == (12, 2, 16.7)
    permit = rows["작업허가 승인 현황"]
    assert (permit.checked, permit.count, permit.rate_pct) == (11, 9, 81.8)
    corrective = rows["개선조치 이행"]
    assert (corrective.checked, corrective.count, corrective.rate_pct) == (6, 3, 50.0)
    missing = rows["측정값 없이 승인된 허가"]
    assert (missing.checked, missing.count, missing.rate_pct) == (None, 1, None)


def test_normal_sentences_have_expected_numbers():
    r = compose(load("normal"))
    by = items_by_type(r)
    risk_text = " ".join(s.text for s in by["risk_assessment_rate"].sentences)
    assert "작업 12건 중 위험성평가를 작성한 작업은 10건(83.3%), 승인된 작업은 8건(66.7%)" in risk_text
    assert "위험성평가 기록이 없는 작업은 2건" in risk_text
    assert "제36조" in risk_text and "법정 의무" in risk_text
    permit_text = " ".join(s.text for s in by["permit_approval_statistics"].sentences)
    assert "승인 요청은 13건이며, 승인 9건, 반려 2건, 취소 1건" in permit_text
    assert "승인률(승인÷(승인+반려))은 81.8%" in permit_text
    assert "차단된 허가는 3건" in permit_text
    action_text = " ".join(s.text for s in by["corrective_action_rate"].sentences)
    assert "6건 중 완료(검증 포함) 3건(50.0%), 검증 완료 1건" in action_text
    assert "미완료 개선조치는 3건이며 이 중 기한이 지난 조치는 1건" in action_text
    assert by["corrective_action_rate"].action_status.overdue == 1
    assert by["risk_assessment_rate"].kind == "LEGAL"
    assert by["permit_approval_statistics"].kind == "INTERNAL"


def test_record_sentences_carry_evidence():
    r = compose(load("normal"))
    by = items_by_type(r)
    missing = [s for s in by["missing_measurements"].sentences if "측정값이 없는 허가는 1건" in s.text]
    assert [(e.type, e.id) for e in missing[0].evidence] == [("PERMIT", 102)]
    blocked = [s for s in by["blocked_records"].sentences if "차단된 허가 기록은 3건" in s.text]
    assert [e.id for e in blocked[0].evidence] == [101, 102, 103]
    actions = [s for s in by["corrective_action_rate"].sentences if "6건 중" in s.text]
    assert len(actions[0].evidence) == 6


def test_record_lines_are_korean_and_short():
    r = compose(load("normal"))
    by = items_by_type(r)
    assert by["risk_assessment_rate"].records.lines[1] == "밀폐공간작업: 작업 4건 중 평가 작성 3건(75.0%)"
    assert by["permit_approval_statistics"].records.lines[0] == "화기작업 허가: 허가 6건, 승인 4건, 반려 1건, 차단 2건"
    assert by["missing_measurements"].records.lines == [
        "허가 CP-202610-0002(밀폐공간작업 허가) · 작업 저장 탱크 내부 점검 · 장소 3구역 탱크동 · 승인일 2026-10-14 · 누락 측정 항목 2개"
    ]
    blocked_lines = by["blocked_records"].records.lines
    assert blocked_lines[1] == "허가 CP-202610-0002(밀폐공간작업 허가) · 차단일 2026-10-14 · 차단 사유 2건 · 산소 농도가 기준 범위를 벗어남"
    action_lines = by["corrective_action_rate"].records.lines
    assert action_lines[0] == "밀폐공간 환기 설비 점검 · 기한 2026-10-10 · 검증 완료 · 출처 점검"
    assert action_lines[4] == "작업허가 서식 안내 교육 · 기한 2026-11-30 · 미완료 · 출처 기타"
    assert by["missing_measurements"].records.shown == 1 and by["missing_measurements"].records.total == 1


def test_overview_scope_and_group_two_unavailable():
    r = compose(load("normal"))
    s = r.overview.scope
    assert (s.works, s.permits, s.near_misses, s.corrective_actions) == (12, 14, None, 6)
    assert r.overall.counts == s
    assert [len(g.items) for g in r.overview.groups] == [2, 2, 3]
    by = items_by_type(r)
    for q in ("near_miss_receipt", "near_miss_action_link"):
        assert by[q].state == "UNAVAILABLE" and by[q].unavailable_reason == "NOT_SUPPORTED"
        assert by[q].state_label == "확인 불가"
        assert by[q].facts.values == []
    assert "GROUP_NOT_SUPPORTED" in codes(r)
    assert r.status == "COMPLETE"  # 미지원 묶음만으로는 불완전이 아니다


def test_overall_lines_review_links_reference():
    r = compose(load("normal"))
    o = r.overall
    assert o.lines[0] == (
        "점검 기간은 2026-10-01부터 2026-10-31까지이며, 조회한 기록은 작업 12건, 작업허가 14건, 개선조치 6건임. "
        "아차사고는 확인하지 못함."
    )
    assert "위험성평가·작업허가: 확인 필요 1개 항목, 현황 참고 1개 항목, 확인 불가 0개 항목임." in o.lines
    assert "아차사고 내역: 확인 필요 0개 항목, 현황 참고 0개 항목, 확인 불가 2개 항목임." in o.lines
    assert "그 밖의 안전관리 기록: 확인 필요 2개 항목, 현황 참고 1개 항목, 확인 불가 0개 항목임." in o.lines
    review = {x.area: x for x in o.review_needed}
    assert review["위험성평가 실시"].count == 2
    assert review["측정값 없이 승인된 허가"].record_ids == [102]
    assert review["측정값 없이 승인된 허가"].record_labels == ["CP-202610-0002"]
    assert review["개선조치 이행"].count == 1 and review["개선조치 이행"].text == "기한이 지난 개선조치 1건"
    assert [x.text for x in o.links] == ["측정값 없이 승인된 허가 1건 중 같은 기간 규칙 차단 기록이 있는 허가는 1건임."]
    assert [(e.type, e.id) for e in o.links[0].evidence] == [("PERMIT", 102)]
    assert [x.text for x in o.reference] == [
        "작업허가 14건 중 승인 9건, 반려 2건, 차단 3건임.",
        "규칙 판정으로 차단된 허가 기록 3건임.",
    ]
    assert o.human_tasks[-1] == "작성자·확인자 서명"
    assert any("확인 불가 항목(아차사고 접수 현황, 아차사고 조치 연결 현황)" in t for t in o.human_tasks)


def test_link_requires_shared_permit_id():
    data = load("normal")
    data["items"][4]["records"][0]["permitId"] = 999  # 측정값 누락 허가가 차단 기록에 없는 허가가 됨
    r = compose(data)
    assert r.overall.links == []


def test_no_evaluative_words_or_internal_codes_in_text():
    for name in ("normal", "zero", "truncated", "unavailable"):
        r = compose(load(name))
        md = render_markdown(r)
        body = md.split("검증 메모")[0]
        for word in ("위반", "부적정", "부적합", "queryType", "stats"):
            assert word not in body, (name, word)
        # 영어 코드(대문자 3자 이상 연속)가 본문에 없어야 한다
        assert not re.search(r"[A-Z]{3,}", body), (name, re.findall(r"[A-Z]{3,}", body))
        for g in r.groups:
            for item in g.items:
                for s in item.sentences:
                    assert not re.search(r"[A-Za-z_]{4,}", s.text), s.text


# ---------------------------------------------------------------- 0건


def test_zero_counts_are_reported_as_zero_not_omitted():
    r = compose(load("zero"))
    by = items_by_type(r)
    assert all(by[q].state == "OK" for q in (
        "risk_assessment_rate", "permit_approval_statistics", "missing_measurements", "blocked_records", "corrective_action_rate"))
    assert [s.text for s in by["risk_assessment_rate"].sentences][-1] == "해당 기간 확인 대상 작업은 0건임."
    assert [s.text for s in by["permit_approval_statistics"].sentences][-1] == "해당 기간 작업허가는 0건임."
    assert [s.text for s in by["missing_measurements"].sentences][1:] == [
        "해당 기간 승인된 허가 중 작업 전 측정값이 없는 허가는 0건임."
    ]
    assert [s.text for s in by["blocked_records"].sentences][1:] == ["해당 기간 규칙 판정으로 차단된 허가 기록은 0건임."]
    assert [s.text for s in by["corrective_action_rate"].sentences][-1] == "해당 기간 등록된 개선조치는 0건임."
    assert r.overall.review_needed == [] and r.overall.links == [] and r.overall.reference == []
    assert r.overview.scope.works == 0 and r.overview.scope.permits == 0 and r.overview.scope.corrective_actions == 0
    risk_row = next(row for g in r.summary.groups for row in g.rows if row.area == "위험성평가 실시")
    assert (risk_row.checked, risk_row.count, risk_row.rate_pct) == (0, 0, None)
    assert r.status == "COMPLETE"


# ---------------------------------------------------------------- 잘림


def test_truncated_list_uses_count_and_marks_total():
    r = compose(load("truncated"))
    m = items_by_type(r)["missing_measurements"]
    assert m.facts.truncated is True and m.facts.total == 130 and m.facts.shown == 100
    assert (m.records.shown, m.records.total, len(m.records.lines)) == (100, 130, 100)
    texts = [s.text for s in m.sentences]
    assert "해당 기간 승인된 허가 중 작업 전 측정값이 없는 허가는 130건임." in texts
    assert "전체 130건 중 100건만 표시함." in texts
    assert "TRUNCATED" in codes(r) and "TRUNCATED_MISMATCH" not in codes(r)
    # 목록이 잘리면 id 교집합을 믿을 수 없어 연결을 만들지 않는다
    assert r.overall.links == [] and "LINK_SKIPPED_TRUNCATED" in codes(r)
    review = next(x for x in r.overall.review_needed if x.area == "측정값 없이 승인된 허가")
    assert review.count == 130 and len(review.record_ids) == 100


def test_truncated_flag_is_computed_not_trusted():
    data = load("truncated")
    data["items"][4]["truncated"] = False
    r = compose(data)
    assert items_by_type(r)["missing_measurements"].facts.truncated is True
    assert "TRUNCATED_MISMATCH" in codes(r)
    data = load("normal")
    data["items"][4]["truncated"] = True
    r = compose(data)
    assert items_by_type(r)["missing_measurements"].facts.truncated is False
    assert "TRUNCATED_MISMATCH" in codes(r)


def test_total_mismatch_warns_and_uses_stats_count():
    data = load("normal")
    data["items"][3]["total"] = 9
    r = compose(data)
    assert "TOTAL_MISMATCH" in codes(r)
    assert items_by_type(r)["blocked_records"].facts.total == 3


def test_records_more_than_count_is_unavailable():
    data = load("normal")
    data["items"][4]["stats"] = {"count": 0}  # 기록은 1건인데 건수 0
    r = compose(data)
    m = items_by_type(r)["missing_measurements"]
    assert m.state == "UNAVAILABLE" and m.unavailable_reason == "INVALID"


# ---------------------------------------------------------------- 확인 불가 (0건으로 바꾸지 않는다)


def test_failures_become_unavailable_never_zero():
    r = compose(load("unavailable"))
    by = items_by_type(r)
    assert r.status == "INCOMPLETE"
    assert by["risk_assessment_rate"].state == "UNAVAILABLE" and by["risk_assessment_rate"].unavailable_reason == "NO_DATA"
    assert by["permit_approval_statistics"].unavailable_reason == "INVALID"
    assert by["blocked_records"].unavailable_reason == "NO_DATA"
    assert by["missing_measurements"].unavailable_reason == "INVALID"
    assert by["corrective_action_rate"].state == "REVIEW"  # 정상 항목은 그대로
    for q in ("risk_assessment_rate", "permit_approval_statistics", "blocked_records", "missing_measurements"):
        item = by[q]
        assert item.facts.values == [] and item.facts.numerator is None and item.facts.total is None
        assert item.records.lines == []
        assert not numbers_in(" ".join(s.text for s in item.sentences)), item.sentences
        assert "0건" not in " ".join(s.text for s in item.sentences)
    s = r.overview.scope
    assert s.works is None and s.permits is None and s.corrective_actions == 6
    rows = {row.area: row for g in r.summary.groups for row in g.rows}
    assert (rows["위험성평가 실시"].checked, rows["위험성평가 실시"].count, rows["위험성평가 실시"].rate_pct) == (None, None, None)
    assert rows["위험성평가 실시"].state_label == "확인 불가"
    assert {"ITEM_MISSING", "STATS_INCONSISTENT", "STATS_INVALID"} <= set(codes(r))
    assert r.overall.lines[0].count("확인 불가") == 2  # 작업·작업허가
    assert "작업 확인 불가" in r.overall.lines[0] and "작업허가 확인 불가" in r.overall.lines[0]
    review_areas = {x.area for x in r.overall.review_needed}
    assert review_areas == {"개선조치 이행"}
    assert r.overall.links == []
    assert "LINK_SKIPPED_TRUNCATED" not in codes(r)  # 조회 실패와 잘림은 다른 사유다


def test_item_sent_but_all_missing_items_marked():
    data = load("normal")
    data["items"] = [data["items"][0]]  # 위험성평가만 보냄
    r = compose(data)
    by = items_by_type(r)
    assert by["risk_assessment_rate"].state == "REVIEW"
    for q in ("permit_approval_statistics", "missing_measurements", "blocked_records", "corrective_action_rate"):
        assert by[q].state == "UNAVAILABLE" and by[q].unavailable_reason == "NO_DATA"
    assert r.status == "INCOMPLETE"


def test_unknown_query_type_is_ignored_with_note():
    data = load("normal")
    data["items"].append({"queryType": "failed_inspections", "title": "점검", "stats": {"count": 4}})
    r = compose(data)
    assert "ITEM_IGNORED" in codes(r)
    assert "failed_inspections" not in all_text(r)


# ---------------------------------------------------------------- 비율·입력 호환


def test_rates_are_recomputed_and_difference_warned():
    data = load("normal")
    data["items"][0]["stats"]["rate"] = 90.0
    r = compose(data)
    assert "RATE_DIFFERS" in codes(r)
    assert "위험성평가를 작성한 작업은 10건(83.3%)" in " ".join(s.text for s in items_by_type(r)["risk_assessment_rate"].sentences)
    assert "RATE_DIFFERS" not in codes(compose(load("normal")))


def test_permit_without_decisions_has_no_rate():
    data = load("normal")
    data["items"][1]["stats"] = {"permits": 2, "requested": 2, "approved": 0, "rejected": 0, "canceled": 0,
                                 "blocked": 1, "approvalRate": None}
    r = compose(data)
    permit = items_by_type(r)["permit_approval_statistics"]
    assert "승인·반려 처리가 없어 승인률은 산정하지 않음." in [s.text for s in permit.sentences]
    assert permit.facts.rate_pct is None and permit.facts.denominator == 0
    assert "RATE_DIFFERS" not in codes(r)


def test_spring_extra_fields_and_totals_are_accepted():
    plain = compose(load("normal"))
    data = load("normal")
    for item in data["items"]:
        item["summary"] = ""
        item["verification"] = None
    for item in data["items"]:
        records = item.get("records") or []
        if item["queryType"] in ("missing_measurements", "blocked_records"):
            item["total"], item["truncated"] = len(records), False
    data["dataSnapshot"] = {"x": 1}
    with_totals = compose(data)
    assert with_totals == plain


def test_request_validation():
    data = load("normal")
    dup = copy.deepcopy(data)
    dup["items"].append(copy.deepcopy(dup["items"][0]))
    with pytest.raises(ValidationError):
        AuditComposeInput.model_validate(dup)
    bad = {**data, "dateFrom": "2026-11-01", "dateTo": "2026-10-01"}
    with pytest.raises(ValidationError):
        AuditComposeInput.model_validate(bad)
    with pytest.raises(ValidationError):
        AuditComposeInput.model_validate({**data, "dateFrom": "2026/10/01"})
    with pytest.raises(ValidationError):
        AuditComposeInput.model_validate({**data, "items": []})
    with pytest.raises(ValidationError):
        AuditComposeInput.model_validate({**data, "siteId": 0})


def test_unknown_codes_and_garbage_rows_do_not_crash():
    data = load("normal")
    data["items"][0]["records"] = [{"workType": "WELDING_X", "works": 4, "written": 3}, {"works": "x"}, {}]
    data["items"][2]["records"] = [{"title": None, "status": "WEIRD", "sourceType": "NEW_SOURCE"}, {}]
    data["items"][3]["records"] = [{"permitId": "7", "permitType": "NEW", "violations": "bad"}]
    data["items"][3]["stats"] = {"count": 1}
    r = compose(data)
    by = items_by_type(r)
    assert by["risk_assessment_rate"].records.lines == ["기타 작업: 작업 4건 중 평가 작성 3건(75.0%)"]
    assert by["corrective_action_rate"].records.lines[0] == "개선조치 · 미완료 · 출처 기타"
    assert by["blocked_records"].records.lines == ["작업허가"]
    assert not re.search(r"WELDING|WEIRD|NEW", " ".join(
        line for item in by.values() for line in item.records.lines))


def test_long_free_text_is_clipped():
    data = load("normal")
    data["items"][4]["records"][0]["workTitle"] = "가" * 200
    line = items_by_type(compose(data))["missing_measurements"].records.lines[0]
    assert "가" * 40 not in line
    assert "가" * 39 + "…" in line


# ---------------------------------------------------------------- 숫자 대조기 동작

def test_sentence_with_wrong_number_is_replaced_with_facts(monkeypatch):
    monkeypatch.setattr(audit_compose, "basis_sentence", lambda spec: "점검 대상은 999건임.")
    r = compose(load("normal"))
    risk = items_by_type(r)["risk_assessment_rate"]
    assert "999" not in all_text(r)
    assert "NARRATIVE_REPLACED" in codes(r)
    assert risk.sentences[0].text == (
        "작업 12건, 평가 작성 10건, 작성 비율 83.3%, 승인 8건, 승인 비율 66.7%, 평가 기록 없음 2건임."
    )
    # 값 나열이 없는 확인 불가 항목에는 일반 문장이 들어간다
    near = items_by_type(r)["near_miss_receipt"]
    assert "999" not in near.sentences[0].text


def test_overall_sentence_with_foreign_number_is_dropped():
    r = compose(load("normal"))
    built_allowed = {"12"}
    assert audit_compose._bad("작업 77건임.", built_allowed, audit_compose._Run()) is True
    assert audit_compose._bad("작업 12건임.", built_allowed, audit_compose._Run()) is False


# ---------------------------------------------------------------- 재현성·독립성


def test_same_input_same_output_except_time():
    a = compose_audit_report(AuditComposeInput.model_validate(load("normal")), now=NOW)
    b = compose_audit_report(AuditComposeInput.model_validate(load("normal")), now=NOW + timedelta(hours=3))
    da, db = a.model_dump(by_alias=True), b.model_dump(by_alias=True)
    assert da["meta"].pop("generatedAt") != db["meta"].pop("generatedAt")
    assert da == db


def test_input_is_not_mutated():
    data = load("normal")
    request = AuditComposeInput.model_validate(data)
    before = request.model_dump()
    compose_audit_report(request, now=NOW)
    assert request.model_dump() == before


def test_composer_does_not_depend_on_llm_or_existing_audit_report():
    root = Path(audit_compose.__file__).parent
    for path in root.glob("audit_compose*.py"):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("app.agent.audit_report", "app.llm", "app.schemas.audit_report", "llm_provider", "requests", "httpx"):
            assert forbidden not in source, (path.name, forbidden)


# ---------------------------------------------------------------- 라우트·미리보기


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_route_returns_camel_case_data(client):
    resp = client.post("/api/v1/agent/audit-report/compose", json=load("normal"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    data = body["data"]
    assert data["formType"] == "AUDIT_REPORT" and data["status"] == "COMPLETE"
    item = data["groups"][0]["items"][0]
    assert item["queryType"] == "risk_assessment_rate" and item["stateLabel"] == "확인 필요"
    assert data["groups"][0]["responderOpinion"] is None
    assert data["overall"]["reviewNeeded"][1]["recordLabels"] == ["CP-202610-0002"]
    assert data["summary"]["groups"][0]["rows"][0]["ratePct"] == 16.7


def test_route_rejects_bad_body(client):
    resp = client.post("/api/v1/agent/audit-report/compose", json={"reportId": 1})
    assert resp.status_code == 422


def test_preview_markdown_has_all_chapters():
    md = render_markdown(compose(load("normal")))
    for heading in ("## Ⅰ. 점검 개요", "## Ⅱ. 점검 결과 요약", "## Ⅲ. 항목별 점검 결과", "## Ⅳ. 종합 내역", "## Ⅴ. 작성·확인"):
        assert heading in md
    assert "(허가 번호 CP-202610-0002)" in md
    assert "작성자 의견: (작성 전)" in md
    md_bad = render_markdown(compose(load("unavailable")))
    assert "확인 불가 항목이 있어 보고서가 완전하지 않습니다." in md_bad


def test_preview_main_prints_markdown_and_json(capsys):
    path = str(FIXTURES / "normal.json")
    assert preview_main([path]) == 0
    assert "## Ⅳ. 종합 내역" in capsys.readouterr().out
    assert preview_main([path, "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["formType"] == "AUDIT_REPORT"


# ---------------------------------------------------------------- 추가 경계 (변형 검사에서 살아남은 것들)


def test_corrective_open_without_overdue_is_reference():
    data = load("normal")
    data["items"][2]["stats"] = {"actions": 6, "completed": 5, "verified": 1, "open": 1, "overdue": 0, "completionRate": 83.3}
    r = compose(data)
    action = items_by_type(r)["corrective_action_rate"]
    assert action.state == "REFERENCE"
    assert [x.text for x in r.overall.reference][-1] == "미완료 개선조치 1건임."
    assert r.overall.review_needed[-1].area != "개선조치 이행"


def test_date_part_and_status_defaults():
    assert date_part("2026-10-14T09:30:00") == "2026-10-14"
    assert date_part("2026-1014abcdef") is None
    assert date_part("2026/10/14 09:30") is None
    assert date_part(20261014) is None
    assert action_status_ko(None) == "미완료"
    assert action_status_ko("WEIRD") == "미완료"
    assert action_status_ko("DONE") == "완료"
    assert action_status_ko("VERIFIED") == "검증 완료"
    assert action_status_ko("CANCELED") == "취소"


def test_link_count_is_added_to_allowed_numbers():
    request = AuditComposeInput.model_validate(load("normal"))
    by = {i.query_type: i for i in request.items}
    run = audit_compose._Run()
    built = {
        "missing_measurements": audit_compose._build_missing(SPECS["missing_measurements"], by["missing_measurements"], 5, run),
        "blocked_records": audit_compose._build_blocked(SPECS["blocked_records"], by["blocked_records"], 6, run),
    }
    allowed: set[str] = set()
    lines = audit_compose._links(built, allowed, run)
    assert len(lines) == 1 and "1" in allowed


def test_evidence_refs_skip_bad_entries_and_dedupe():
    data = load("normal")
    data["items"][4]["evidence"] = [
        {"type": "PERMIT", "id": True},
        {"type": "PERMIT", "id": 102},
        {"type": "PERMIT", "id": 102},
        {"type": 5, "id": 1},
        {"type": "PERMIT"},
    ]
    r = compose(data)
    sentence = next(s for s in items_by_type(r)["missing_measurements"].sentences if "1건임" in s.text)
    assert [(e.type, e.id) for e in sentence.evidence] == [("PERMIT", 102)]


def test_bool_counts_in_record_rows_are_ignored():
    data = load("normal")
    data["items"][0]["records"] = [{"workType": "HOT", "works": True, "written": True}]
    assert items_by_type(compose(data))["risk_assessment_rate"].records.lines == []


def test_group_number_uses_start_month():
    data = load("normal")
    data["dateFrom"], data["dateTo"] = "2026-09-15", "2026-10-14"
    assert compose(data).groups[0].no == "AR-1-202609-01"


def test_review_record_labels_are_deduped():
    data = load("normal")
    row = copy.deepcopy(data["items"][4]["records"][0])
    row["permitId"] = 777
    data["items"][4]["records"].append(row)
    data["items"][4]["stats"] = {"count": 2}
    review = next(x for x in compose(data).overall.review_needed if x.area == "측정값 없이 승인된 허가")
    assert review.record_ids == [102, 777] and review.record_labels == ["CP-202610-0002"]


def test_blocked_truncation_marks_sentence_and_skips_links():
    data = load("normal")
    blocked = data["items"][3]
    blocked["stats"] = {"count": 120}
    blocked["records"] = [
        {"permitId": 200 + i, "permitNo": f"HP-{i:04d}", "permitType": "HOT", "blockedAt": "2026-10-07T09:12:00", "violations": []}
        for i in range(100)
    ]
    blocked["evidence"] = []
    r = compose(data)
    item = items_by_type(r)["blocked_records"]
    assert "전체 120건 중 100건만 표시함." in [s.text for s in item.sentences]
    assert item.facts.truncated is True and item.records.total == 120
    assert r.overall.links == [] and "LINK_SKIPPED_TRUNCATED" in codes(r)


def test_corrective_fact_values_are_exact():
    r = compose(load("normal"))
    values = {v.label: (v.value, v.unit) for v in items_by_type(r)["corrective_action_rate"].facts.values}
    assert values == {
        "개선조치": (6, "건"), "완료(검증 포함)": (3, "건"), "조치율": (50.0, "%"),
        "검증 완료": (1, "건"), "미완료": (3, "건"), "기한 경과": (1, "건"),
    }


def test_review_text_with_foreign_number_is_replaced_by_area():
    request = AuditComposeInput.model_validate(load("normal"))
    run = audit_compose._Run()
    built = audit_compose._build_all(request, run)
    built["risk_assessment_rate"].review.text = "위험성평가 기록이 없는 작업 99건"
    allowed = audit_compose._allowed_numbers(built)
    scope = audit_compose.ComposeScope(works=12)
    overall = audit_compose._build_overall(request, built, scope, allowed, run)
    risk = next(x for x in overall.review_needed if x.area == "위험성평가 실시")
    assert risk.text == "위험성평가 실시"
    assert "NARRATIVE_REPLACED" in [v.code for v in run.verification]


def test_preview_out_file_is_utf8(tmp_path, capsys):
    out = tmp_path / "preview.md"
    assert preview_main([str(FIXTURES / "normal.json"), "--out", str(out)]) == 0
    assert out.read_bytes().startswith("# 안전관리".encode("utf-8"))
    assert b"\r\n" not in out.read_bytes()
    assert str(out) in capsys.readouterr().out


def test_group_state_counts_are_never_dropped_from_overall():
    """상태별 항목 수는 묶음마다 다르므로 허용 숫자에 들어 있어야 한다(확인 불가가 많은 경우에도 줄이 남는다)."""
    for name in ("normal", "zero", "truncated", "unavailable"):
        r = compose(load(name))
        assert len(r.overall.lines) == 4, name
        assert "NARRATIVE_REPLACED" not in codes(r), name
    r = compose(load("unavailable"))
    assert "그 밖의 안전관리 기록: 확인 필요 1개 항목, 현황 참고 0개 항목, 확인 불가 2개 항목임." in r.overall.lines
    assert "위험성평가·작업허가: 확인 필요 0개 항목, 현황 참고 0개 항목, 확인 불가 2개 항목임." in r.overall.lines


def test_preview_caps_long_label_lists():
    md = render_markdown(compose(load("truncated")))
    assert "CP-202610-0009" in md and "CP-202610-0010" not in md.split("검증 메모")[0].split("### 2.")[1].split("### 3.")[0]
    assert "외 90건" in md
