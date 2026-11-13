"""DONE 뒤의 후속 말 처리 시험 (app/agent/followup.py, text_rules.REPLACE_NEGATION).

스텁 하니스: mock 도구 + FakeLLMClient로 WorkRegisterAgent를 실제로 돌린다 (모델·RAG·Spring 없음).
계획서 9장 시나리오 A 두 턴으로 DONE 상태를 만든 뒤, 다음 말이 어떻게 처리되는지 본다.
"""

from __future__ import annotations

import json

import pytest

from app.agent import followup, text_rules
from app.agent.slots import merge_slots
from app.agent.work_register import WorkRegisterAgent
from app.llm.llm_fake import FakeLLMClient, text_response
from app.schemas.agent_work import WorkRegisterRequest, WorkSlots
from app.tools.tool_setup import build_tool_registry

FIRST = "오늘 13시에 화기작업 있어. 필요한 거 만들어줘."
SECOND = (
    "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 옆 T-105 탱크에 톨루엔. 3명, 15시까지. 크레인은 안 씀."
)
_PLAN = json.dumps(
    {"summary": "R-201 반응기 하부 배관 플랜지 볼트 교체", "equipment": ["그라인더"], "materials": ["톨루엔"]}, ensure_ascii=False
)


def make_agent() -> WorkRegisterAgent:
    return WorkRegisterAgent(llm=FakeLLMClient([text_response(_PLAN)] * 60), tools=build_tool_registry())


def run(agent: WorkRegisterAgent, message: str, slots: WorkSlots | None = None):
    return agent.run(WorkRegisterRequest(message=message, slots=slots, run_id=1, site_id=1))


@pytest.fixture
def done():
    """(agent, DONE 응답). 대본 두 턴."""
    agent = make_agent()
    first = run(agent, FIRST)
    assert first.run_status == "WAITING_INPUT"
    second = run(agent, SECOND, first.slots)
    assert second.run_status == "DONE"
    return agent, second


def tool_names(result) -> list[str]:
    return [s.tool_name for s in result.steps if s.kind == "TOOL"]


# ── 대본 두 턴은 변경 전과 같다 (DONE 뒤가 아니면 followup은 끼어들지 않는다) ──

def test_first_two_turns_not_touched():
    agent = make_agent()
    assert followup.judge_followup(None, FIRST) is None
    first = run(agent, FIRST)
    assert first.run_status == "WAITING_INPUT" and first.slots.asked_keys  # 되묻는 중 → followup 없음
    assert followup.judge_followup(first.slots, SECOND) is None
    assert not first.ask.message.startswith("변경") and "새 작업" not in first.ask.message


def test_done_state_detection(done):
    _, result = done
    assert followup.is_done_state(result.slots)
    assert not followup.is_done_state(None)
    assert not followup.is_done_state(WorkSlots())
    assert not followup.is_done_state(WorkSlots(utterances=["화기작업"], asked_keys=["location"]))


def test_done_state_requires_no_pending_ask_even_when_fields_are_full(done):
    """필수 항목이 다 차 있어도 되묻는 중(askedKeys)·시나리오 되묻기(pendingText)·발화 없음은 DONE 뒤가 아니다."""
    _, result = done
    full = result.slots
    assert followup.is_done_state(full)
    assert not followup.is_done_state(full.model_copy(update={"asked_keys": ["lel"]}))
    assert not followup.is_done_state(full.model_copy(update={"pending_text": "앞선 말"}))
    assert not followup.is_done_state(full.model_copy(update={"utterances": []}))


# ── 무변경: 파이프라인을 돌리지 않는다 ──

@pytest.mark.parametrize("message", ["고마워", "위험요인 설명해줘", "네 확인했습니다"])
def test_unchanged_message_returns_guide_without_pipeline(done, message):
    agent, previous = done
    llm_calls = len(agent._llm.calls)
    result = run(agent, message, previous.slots)
    assert result.run_status == "WAITING_INPUT"
    assert result.ask.message == followup.REPLY_UNCHANGED
    assert result.ask.missing_fields == []
    assert result.slots == previous.slots  # slots 그대로 (발화도 쌓지 않는다)
    assert len(agent._llm.calls) == llm_calls  # 계획 LLM을 부르지 않았다
    assert tool_names(result) == ["classify_work"]  # 검색·MSDS·규칙 판정·양식 작성 없음
    assert result.drafts is None or not result.drafts.permits


def test_unchanged_twice_keeps_done_state(done):
    agent, previous = done
    r1 = run(agent, "고마워", previous.slots)
    r2 = run(agent, "인원 4명으로", r1.slots)  # 안내 뒤에도 이어서 수정할 수 있다
    assert r2.run_status == "DONE" and r2.slots.headcount == 4


def test_change_marker_without_value_asks_what_to_change(done):
    agent, previous = done
    result = run(agent, "시간 좀 바꿔줘", previous.slots)
    assert result.run_status == "WAITING_INPUT" and result.ask.message == followup.REPLY_WHAT_TO_CHANGE


# ── 수정: 병합해 다시 만들고 변경 요약을 붙인다 ──

def test_headcount_edit_regenerates_with_change_note(done):
    agent, previous = done
    result = run(agent, "인원 4명으로 바꿔줘", previous.slots)
    assert result.run_status == "DONE"
    assert result.slots.headcount == 4 and result.slots.location == previous.slots.location
    assert result.ask.message.startswith("변경: 인원 3명 → 4명\n")
    assert result.drafts.risk_assessment.headcount == 4
    assert result.slots.utterances[-1] == "인원 4명으로 바꿔줘"


def test_date_only_change_is_edit(done):
    agent, previous = done
    result = run(agent, "내일로 바꿔줘", previous.slots)
    assert result.run_status == "DONE" and result.slots.work_date != previous.slots.work_date
    assert result.ask.message.startswith("변경: 날짜 ")


def test_time_edit(done):
    agent, previous = done
    result = run(agent, "16시까지로 변경", previous.slots)
    assert result.run_status == "DONE" and result.slots.end_time == "16:00"
    assert "종료 15:00 → 16:00" in result.ask.message


def test_equipment_addition_is_edit_not_new_task(done):
    agent, previous = done
    result = run(agent, "크레인도 쓸게", previous.slots)  # 크레인 → 중량물이지만 장비 이름이라 카테고리 변경이 아니다
    assert result.run_status == "DONE" and "크레인" in result.slots.equipment
    assert result.slots.location == previous.slots.location and result.slots.hot_work_kind == previous.slots.hot_work_kind


def test_fire_watcher_edit_removes_block_reason(done):
    agent, previous = done
    assert any("화재감시자" in r for r in previous.approval_gate.reasons)
    result = run(agent, "화재감시자는 김철수", previous.slots)
    assert result.slots.fire_watcher == "김철수"
    assert not any("화재감시자" in r for r in result.approval_gate.reasons)
    assert result.approval_gate.can_request is False  # LEL 값이 아직 없다 (모름을 통과로 바꾸지 않는다)


def test_material_mention_near_other_tank_is_edit(done):
    agent, previous = done
    result = run(agent, "T-105 탱크 옆에 메탄올도 있어", previous.slots)  # 장소처럼 보이지만 작업 내용이 아니다
    assert result.run_status == "DONE" and "메탄올" in result.slots.nearby_materials
    assert result.slots.location == previous.slots.location


def test_replace_expression_swaps_grinder_for_welding(done):
    agent, previous = done
    result = run(agent, "그라인더 말고 용접으로 할게", previous.slots)
    assert result.run_status == "DONE"
    assert result.slots.hot_work_kind == "용접"
    assert "그라인더" not in result.slots.equipment and "그라인더" in result.slots.excluded_equipment
    hazards = [i.hazard for i in result.drafts.risk_assessment.items]
    assert not any("그라인더" in h for h in hazards)  # 연삭 위험요인이 빠진다
    assert any("흄" in h or "용접" in h for h in hazards)
    assert not any("그라인더" in i.item for i in result.drafts.inspection_checklist.items)
    assert result.ask.message.startswith("변경: ")


def test_regenerate_request_without_change(done):
    agent, previous = done
    result = run(agent, "다시 만들어줘", previous.slots)
    assert result.run_status == "DONE" and result.slots == previous.slots
    assert result.ask.message.startswith(followup.NOTE_REGENERATED)


# ── 새 작업: 기본(guide)은 안내만 하고 이전 slots·초안을 그대로 둔다 ──

NEW_TASK_MESSAGES = [
    "R-201 반응기 하부 배관 안에 들어가서 청소할 거야 밀폐공간 작업",  # 같은 장소, 카테고리 변경
    "R-305 반응기 배관 교체 작업 있어",  # 다른 장소, 수정 표지어 없음
    "내일 9시에 B동 지붕 위 환풍기 교체, 2명, 11시까지. 사다리 사용, 장비 없음, 물질 없음.",  # 정보가 다 찬 새 작업
]


@pytest.mark.parametrize("message", NEW_TASK_MESSAGES)
def test_new_task_default_guides_and_keeps_previous_draft(done, message):
    assert followup.NEW_TASK_MODE == "guide"
    agent, previous = done
    llm_calls = len(agent._llm.calls)
    result = run(agent, message, previous.slots)
    assert result.run_status == "WAITING_INPUT" and result.ask.message == followup.REPLY_NEW_TASK
    assert "새 대화" in result.ask.message and result.ask.missing_fields == []
    assert result.slots == previous.slots  # 이전 slots 그대로: 기존 초안이 사라지지 않는다
    assert len(agent._llm.calls) == llm_calls and tool_names(result) == ["classify_work"]


def test_previous_draft_still_editable_after_new_task_guide(done):
    agent, previous = done
    guided = run(agent, NEW_TASK_MESSAGES[0], previous.slots)
    result = run(agent, "인원 5명으로 바꿔줘", guided.slots)
    assert result.run_status == "DONE" and result.ask.message.startswith("변경: 인원 3명 → 5명")
    assert result.slots.hot_work_kind == previous.slots.hot_work_kind


# ── 선택 모드 reset: 이전 slots를 버리고 새로 시작한다 ──

@pytest.fixture
def reset_mode(monkeypatch):
    monkeypatch.setattr(followup, "NEW_TASK_MODE", "reset")

def test_reset_mode_category_change_same_place_is_new_task(done, reset_mode):
    agent, previous = done
    llm_calls = len(agent._llm.calls)
    result = run(agent, "R-201 반응기 하부 배관 안에 들어가서 청소할 거야 밀폐공간 작업", previous.slots)
    assert result.run_status == "WAITING_INPUT"  # 새 작업이라 필수 항목을 다시 묻는다
    assert result.ask.message.startswith(followup.NOTE_NEW_TASK)
    assert result.slots.hot_work_kind is None and result.slots.nearby_materials == [] and result.slots.headcount is None
    assert result.slots.utterances == ["R-201 반응기 하부 배관 안에 들어가서 청소할 거야 밀폐공간 작업"]
    assert result.classification.work_types == ["밀폐공간"]
    assert len(agent._llm.calls) == llm_calls  # 되묻는 중이라 파이프라인은 안 돈다


def test_reset_mode_new_task_with_everything_goes_straight_to_done(done, reset_mode):
    agent, previous = done
    msg = "내일 9시에 B동 지붕 위 환풍기 교체, 2명, 11시까지. 사다리 사용, 장비 없음, 물질 없음."
    result = run(agent, msg, previous.slots)
    assert result.run_status == "DONE" and result.ask.message.startswith(followup.NOTE_NEW_TASK)
    assert result.classification.work_types == ["고소"] and result.classification.required_permits == ["HEIGHT_WORK_PERMIT"]
    assert result.slots.hot_work_kind is None and result.slots.headcount == 2


def test_reset_mode_different_place_without_marker_is_new_task(done, reset_mode):
    agent, previous = done
    result = run(agent, "R-305 반응기 배관 교체 작업 있어", previous.slots)
    assert result.ask.message.startswith(followup.NOTE_NEW_TASK)
    assert result.slots.location is not None and "R-305" in result.slots.location and result.slots.utterances == ["R-305 반응기 배관 교체 작업 있어"]


def test_different_place_with_marker_is_edit(done):
    agent, previous = done
    result = run(agent, "장소를 R-305 반응기 하부 배관으로 바꿔줘", previous.slots)
    assert result.run_status == "DONE" and "R-305" in (result.slots.location or "")
    assert result.ask.message.startswith("변경: 장소 ")


@pytest.mark.parametrize("message", ["R-202로 바꿔줘", "R-202"])
def test_place_only_message_is_edit(done, message):
    agent, previous = done
    result = run(agent, message, previous.slots)
    assert result.run_status == "DONE" and "R-202" in (result.slots.location or "")
    assert result.ask.message.startswith("변경: 장소 ")
    assert not result.ask.message.startswith(followup.NOTE_NEW_TASK)


def test_reset_mode_guide_reply_not_used(done, reset_mode):
    agent, previous = done
    result = run(agent, "탱크 내부 청소하는 밀폐공간 작업 있어", previous.slots)
    assert result.ask.message != followup.REPLY_NEW_TASK and result.slots != previous.slots


# ── 끄면 변경 전과 같다 ──

def test_followup_disabled_regenerates_like_before(done, monkeypatch):
    monkeypatch.setattr(followup, "FOLLOWUP_ENABLED", False)
    agent, previous = done
    result = run(agent, "고마워", previous.slots)
    assert result.run_status == "DONE"  # 예전 동작: 무엇을 말해도 초안을 다시 만든다
    assert not result.ask.message.startswith("변경")
    assert result.slots.utterances[-1] == "고마워"
    mixed = run(agent, "탱크 내부 청소하는 밀폐공간 작업 있어", previous.slots)
    assert mixed.classification.work_types == ["화기", "밀폐공간"]  # 예전 동작: 이전 값과 섞인다


# ── 교체 표현 (text_rules) ──

def test_replace_words_negate_previous_term():
    assert text_rules.is_negated("그라인더 말고 용접", len("그라인더"))
    assert text_rules.is_negated("그라인더 대신 용접", len("그라인더"))
    assert text_rules.is_negated("그라인더가 아니라 용접", len("그라인더"))
    assert not text_rules.is_negated("그라인더 말고도 용접기를 씁니다", len("그라인더"))  # '말고도'는 그 밖에도
    assert text_rules.is_negated("크레인은 안 씀", len("크레인"))  # 기존 표현은 그대로


def test_replace_negation_switch_restores_old_pattern():
    """REPLACE_NEGATION = False로 소스를 바꿔 새 이름공간에서 실행하면 변경 전 패턴이다 (다른 모듈에 영향 없음)."""
    from pathlib import Path

    src = Path(text_rules.__file__).read_text(encoding="utf-8")
    assert "REPLACE_NEGATION = True" in src
    off: dict = {}
    exec(compile(src.replace("REPLACE_NEGATION = True", "REPLACE_NEGATION = False"), "text_rules_off", "exec"), off)
    assert not off["is_negated"]("그라인더 말고 용접", len("그라인더"))
    assert not off["is_negated"]("그라인더 대신 용접", len("그라인더"))
    assert off["is_negated"]("크레인은 안 씀", len("크레인"))


def test_replace_expression_in_slots_extraction():
    slots = merge_slots(WorkSlots(utterances=["그라인더 절단"], hot_work_kind="그라인더 절단", equipment=["그라인더"]), "그라인더 말고 용접으로 할게")
    assert slots.hot_work_kind == "용접" and slots.equipment == [] and slots.excluded_equipment == ["그라인더"]
