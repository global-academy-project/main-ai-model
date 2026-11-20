"""구성기 응답의 종합 의견 초안(`overall.draftText`) 시험.

화면은 종합 의견 칸이 비어 있을 때 이 값을 미리 넣는다 (Spring은 응답을 그대로 `compose_json`에 둔다).
확인할 것: ① 초안은 종합 내역의 문장만 줄바꿈으로 잇는다(새 문장 없음) ② 사람이 할 일 목록은 넣지 않는다
③ 숫자 대조에서 빠진 문장은 초안에도 없다 ④ 끄면 빈 문자열이고 나머지 응답은 같다 ⑤ JSON 키 draftText.
`.env`·터미널 환경에 기대지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import audit_compose
from app.agent.audit_compose import (
    DRAFT_REFERENCE_PREFIX,
    DRAFT_REVIEW_PREFIX,
    compose_audit_report,
)
from app.core import get_settings
from app.schemas.audit_compose import AuditComposeInput

FIXTURES = Path(__file__).parent / "fixtures" / "audit_compose"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def compose(name: str):
    return compose_audit_report(AuditComposeInput.model_validate(load(name)))


def expected_draft(overall) -> str:
    parts = list(overall.lines)
    parts += [DRAFT_REVIEW_PREFIX + r.text for r in overall.review_needed]
    parts += [DRAFT_REFERENCE_PREFIX + o.text for o in overall.reference]
    parts += [o.text for o in overall.links]
    return "\n".join(parts)


# ---------------------------------------------------------------- 내용


@pytest.mark.parametrize("name", ["normal", "zero", "unavailable", "truncated"])
def test_draft_is_the_overall_sentences_joined_in_order(name):
    overall = compose(name).overall
    assert overall.draft_text == expected_draft(overall)
    assert overall.draft_text != ""


def test_known_sample_draft_text():
    """정상 입력의 초안 전체를 고정한다 (순서: 요약 줄 → 확인 필요 → 현황 참고 → 항목 간 연결)."""
    overall = compose("normal").overall
    lines = overall.draft_text.split("\n")
    assert lines[: len(overall.lines)] == overall.lines
    rest = lines[len(overall.lines) :]
    assert [x for x in rest if x.startswith(DRAFT_REVIEW_PREFIX)] == [DRAFT_REVIEW_PREFIX + r.text for r in overall.review_needed]
    assert [x for x in rest if x.startswith(DRAFT_REFERENCE_PREFIX)] == [DRAFT_REFERENCE_PREFIX + o.text for o in overall.reference]
    order = [0 if x.startswith(DRAFT_REVIEW_PREFIX) else 1 if x.startswith(DRAFT_REFERENCE_PREFIX) else 2 for x in rest]
    assert order == sorted(order)


def test_human_tasks_are_not_in_the_draft():
    overall = compose("normal").overall
    assert overall.human_tasks
    for task in overall.human_tasks:
        assert task not in overall.draft_text
    assert "서명" not in overall.draft_text


def test_every_draft_line_comes_from_the_overall_lists():
    overall = compose("normal").overall
    allowed = set(overall.lines)
    allowed |= {DRAFT_REVIEW_PREFIX + r.text for r in overall.review_needed}
    allowed |= {DRAFT_REFERENCE_PREFIX + o.text for o in overall.reference}
    allowed |= {o.text for o in overall.links}
    for line in overall.draft_text.split("\n"):
        assert line in allowed


def test_draft_has_no_blank_lines_and_no_placeholders():
    for name in ("normal", "zero", "unavailable", "truncated"):
        text = compose(name).overall.draft_text
        assert all(line.strip() for line in text.split("\n"))
        assert "None" not in text and "null" not in text


def test_unavailable_item_stays_unavailable_in_the_draft():
    text = compose("unavailable").overall.draft_text
    assert "확인 불가" in text


def test_zero_counts_are_written_as_zero_not_dropped():
    text = compose("zero").overall.draft_text
    assert "0건" in text or "0개" in text


# ---------------------------------------------------------------- 숫자 대조에서 빠진 문장


def test_a_sentence_dropped_by_the_number_check_is_not_in_the_draft(monkeypatch):
    before = compose("normal").overall
    victim = next(o.text for o in before.reference)
    assert victim in before.draft_text
    real = audit_compose._bad
    monkeypatch.setattr(audit_compose, "_bad", lambda text, allowed, run: text == victim or real(text, allowed, run))
    after = compose("normal").overall
    assert victim not in [o.text for o in after.reference]
    assert victim not in after.draft_text


def test_a_dropped_summary_line_is_not_in_the_draft(monkeypatch):
    before = compose("normal").overall
    victim = before.lines[1]
    real = audit_compose._bad
    monkeypatch.setattr(audit_compose, "_bad", lambda text, allowed, run: text == victim or real(text, allowed, run))
    after = compose("normal").overall
    assert victim not in after.lines
    assert victim not in after.draft_text


# ---------------------------------------------------------------- 끄기


@pytest.mark.parametrize("name", ["normal", "zero", "unavailable", "truncated"])
def test_switch_off_leaves_an_empty_draft_and_the_rest_unchanged(monkeypatch, name):
    def dump(result) -> dict:
        data = result.model_dump(mode="json", by_alias=True)
        data["meta"]["generatedAt"] = None
        return data

    on = dump(compose(name))
    monkeypatch.setattr(audit_compose, "DRAFT_TEXT", False)
    off = dump(compose(name))
    assert off["overall"]["draftText"] == ""
    assert on["overall"].pop("draftText") != ""
    off["overall"].pop("draftText")
    assert on == off


def test_same_input_gives_the_same_draft():
    assert compose("normal").overall.draft_text == compose("normal").overall.draft_text


# ---------------------------------------------------------------- 응답 모양


def test_json_key_is_draft_text_in_camel_case():
    data = compose("normal").model_dump(mode="json", by_alias=True)
    assert "draftText" in data["overall"]
    assert "draft_text" not in data["overall"]
    # 다른 필드는 그대로 있다 (Spring은 모르는 필드를 그대로 보관한다)
    assert {"counts", "lines", "reviewNeeded", "links", "reference", "humanTasks"} <= set(data["overall"])


def test_http_response_carries_the_draft_text(monkeypatch, tmp_path):
    from app.main import create_app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCENARIO_ENGINES", '{"default": {"engine": "FAKE"}}')
    monkeypatch.setenv("AUDIT_NARRATIVE_LLM", "false")
    get_settings.cache_clear()
    try:
        app: FastAPI = create_app()
        response = TestClient(app).post("/api/v1/agent/audit-report", json=load("normal"))
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200, response.text
    overall = response.json()["data"]["overall"]
    assert overall["draftText"] == "\n".join(
        overall["lines"]
        + [DRAFT_REVIEW_PREFIX + r["text"] for r in overall["reviewNeeded"]]
        + [DRAFT_REFERENCE_PREFIX + o["text"] for o in overall["reference"]]
        + [o["text"] for o in overall["links"]]
    )
