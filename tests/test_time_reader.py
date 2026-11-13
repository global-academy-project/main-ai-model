"""시각 읽기 개선 시험 (app/agent/time_reader.py + slots.py 연결, 11/13).

스텁 하니스: 실제 모델·RAG·Spring 없이 slots.py의 읽기·병합·되묻기 문구를 직접 시험한다. 표는 설계 문서 6장과 같다
(`claude/시각읽기_개선_설계_v0.md`). 기대값은 11/05 사용자 규칙(주간 작업 기본)과 11/13 결정이다.
"""

from __future__ import annotations

import pytest

from app.agent import slots as slots_mod
from app.agent import time_reader
from app.agent.slots import ask_message, extract_slots, merge_slots, missing_fields
from app.schemas.agent_work import WorkSlots


def times(message: str, expecting=()):
    found = extract_slots(message, expecting)
    return found.get("start_time"), found.get("end_time")


def questions(slots: WorkSlots, work_types=("일반",)):
    return {f.key: f.question for f in missing_fields(slots, list(work_types)) if f.key in ("start_time", "end_time")}


def full(**kw) -> WorkSlots:
    """시간만 빼고 필수 항목이 다 찬 slots."""
    base = dict(location="R-201 반응기", headcount=3, equipment_none=True, utterances=["R-201 반응기 점검 3명 장비 없음"])
    base.update(kw)
    return WorkSlots.model_validate(base)


# ── A. 회귀: 변경 전과 같아야 하는 읽기 (11/05 규칙 포함) ──

A_ROWS = [
    ("9시에서 11시까지", "09:00", "11:00"),
    ("9시~11시", "09:00", "11:00"),
    ("9시부터 11시까지", "09:00", "11:00"),
    ("오전 9시부터 오후 2시까지", "09:00", "14:00"),
    ("3시부터 5시까지", "15:00", "17:00"),
    ("오늘 13시에 화기작업 있어", "13:00", None),
    ("11시까지", None, "11:00"),
    ("11시 30분까지", None, "11:30"),
    ("11시 반까지", None, "11:30"),
    ("11:30까지", None, "11:30"),
    ("9시 반에 시작", "09:30", None),
    ("9시 30분부터", "09:30", None),
    ("오전 3시~5시", "03:00", "05:00"),
    ("새벽 3시부터 6시까지", "03:00", "06:00"),
    ("오전 9시부터 5시까지", "09:00", "17:00"),
    ("6시에 시작", "06:00", None),
    ("6시까지", None, "18:00"),
    ("저녁 7시부터 밤 11시까지", "19:00", "23:00"),
    ("16시까지로 바꿔줘", None, "16:00"),
    ("오후 1시~3시", "13:00", "15:00"),
    ("11시부터 1시까지", "11:00", "13:00"),
    ("10~12시", "10:00", "12:00"),
]


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("message,start,end", A_ROWS)
def test_regression_rows_read_the_same_with_switch_on_and_off(monkeypatch, v2, message, start, end):
    monkeypatch.setattr(time_reader, "TIME_READER_V2", v2)
    assert times(message) == (start, end)


@pytest.mark.parametrize(
    "message",
    ["R-201 점검 3명", "고마워", "2시간 동안 작업", "한 시설 점검", "두 시스템 점검", "배열 시 확인 필요", "구성한 시 점검", "1시간 연장"],
)
def test_messages_without_a_time_read_nothing(message):
    assert times(message) == (None, None)
    assert time_reader.read_times(message).problem is None


# ── B. 새로 읽는 말 ──

B_ROWS = [
    ("시작은 9시", "09:00", None),
    ("시작 시간은 9시야", "09:00", None),
    ("9시 시작", "09:00", None),
    ("오전 9시 시작", "09:00", None),
    ("09:00 시작", "09:00", None),
    ("9시경 시작", "09:00", None),
    ("아홉 시에 시작", "09:00", None),
    ("종료는 11시", None, "11:00"),
    ("종료 시간은 11시예요", None, "11:00"),
    ("끝나는 시간은 11시", None, "11:00"),
    ("11시쯤 끝나", None, "11:00"),
    ("11시 전까지", None, "11:00"),
    ("11시 이전까지", None, "11:00"),
    ("11시에 마쳐", None, "11:00"),  # 오독 수정: 예전에는 시작 11:00
    ("11시에 마무리", None, "11:00"),  # 오독 수정
    ("열한 시까지", None, "11:00"),
    ("11시에 용접 작업 마쳐요", None, "11:00"),  # 'N시에'를 시작으로 읽지 않는다 (같은 절에 종료 단서)
    ("9시에 용접 작업 시작해요", "09:00", None),
    ("13시에 시작해서 15시에 끝내요", "13:00", "15:00"),
    ("9시-11시", "09:00", "11:00"),
    ("9시부터 11시", "09:00", "11:00"),
    ("9시부터 11시 사이", "09:00", "11:00"),
    ("15시까지가 아니라 16시까지로", None, "16:00"),  # 오독 수정: 예전에는 종료 15:00
    ("15시가 아니라 16시까지", None, "16:00"),
    ("15시 말고 16시까지", None, "16:00"),
    ("종료 시간을 16시로 바꿔줘", None, "16:00"),
    ("종료를 16시로 수정", None, "16:00"),
    ("시작을 10시로 바꿔줘", "10:00", None),
    ("시작 시간 10시로 변경", "10:00", None),
    ("시작을 10시 말고 11시로", "11:00", None),
    ("9시에 시작해서 11시에 끝나요", "09:00", "11:00"),
    ("시작 9시, 종료 11시", "09:00", "11:00"),
    ("9시 시작 11시 종료", "09:00", "11:00"),
    ("시작 시간은 9시 종료 시간은 11시", "09:00", "11:00"),
    ("작업 시작은 오전 9시, 종료는 오후 3시", "09:00", "15:00"),
    ("3시에서 5시 사이에 할 거야", "15:00", "17:00"),
    ("9시부터 2시간", "09:00", None),  # '2시간'은 시각이 아니다
    ("13시에 시작해서 2시간 정도 걸려요", "13:00", None),
]


@pytest.mark.parametrize("message,start,end", B_ROWS)
def test_new_readings(message, start, end):
    assert times(message) == (start, end)


@pytest.mark.parametrize(
    "message,asked,start,end",
    [
        ("종료는 11시", ["end_time"], None, "11:00"),
        ("11시에 마쳐요", ["end_time"], None, "11:00"),
        ("열한 시", ["end_time"], None, "11:00"),
        ("끝나는 건 11시", ["end_time"], None, "11:00"),
        ("11시", ["end_time"], None, "11:00"),
        ("11시쯤이요", ["end_time"], None, "11:00"),
        ("시작은 9시", ["start_time"], "09:00", None),
        ("9시에 시작해요", ["start_time"], "09:00", None),
        ("아홉 시", ["start_time"], "09:00", None),
        ("9시요", ["start_time"], "09:00", None),
        ("11시에 마쳐요", ["start_time"], None, "11:00"),  # 반대 역할 단서가 있으면 물은 항목의 답으로 보지 않는다
    ],
)
def test_answers_to_a_question_about_one_time(message, asked, start, end):
    assert times(message, asked) == (start, end)


def test_bare_time_is_not_guessed_when_both_were_asked_or_nothing_was_asked():
    assert times("10시", ["start_time", "end_time"]) == (None, None)
    assert times("10시") == (None, None)
    assert times("9시 11시", ["start_time"]) == (None, None)  # 단서 없는 시각이 둘이면 정하지 않는다


# ── C. 되묻기: 오독 위험이면 읽지 않고 비운다 ──

C_UNREAD = ["9시 11시 14시", "9시에 시작해서 10시에 시작", "16시로 변경"]
C_MISMATCH = ["9시부터 9시까지", "6시부터 6시까지", "15시부터 11시까지", "11시부터 9시까지", "오후 2시부터 오전 9시까지", "저녁 7시부터 11시까지"]
C_OVERNIGHT = ["23시부터 2시까지", "밤 10시부터 새벽 2시까지", "저녁 7시부터 새벽 2시까지"]


@pytest.mark.parametrize("message", C_UNREAD)
def test_unreadable_times_are_not_read_and_reason_is_shown(message):
    assert times(message) == (None, None)
    assert time_reader.read_times(message).problem == "unread"
    merged = merge_slots(None, message)
    assert merged.start_time is None and merged.end_time is None
    q = questions(merged)
    assert q["start_time"].startswith(time_reader.PREFIX_UNREAD)


@pytest.mark.parametrize("message", C_MISMATCH)
def test_mismatch_blanks_both_and_asks_for_ampm_or_24h(message):
    merged = merge_slots(None, message)
    assert merged.start_time is None and merged.end_time is None
    for text in questions(merged).values():
        assert text == time_reader.MSG_MISMATCH
    assert "오전·오후" in time_reader.MSG_MISMATCH and "24시간제" in time_reader.MSG_MISMATCH


@pytest.mark.parametrize("message", C_OVERNIGHT)
def test_overnight_is_not_supported_and_guides(message):
    merged = merge_slots(None, message)
    assert merged.start_time is None and merged.end_time is None
    for text in questions(merged).values():
        assert text == time_reader.MSG_OVERNIGHT
    assert "지원하지 않습니다" in time_reader.MSG_OVERNIGHT


def test_same_role_with_two_values_asks_only_that_role():
    merged = merge_slots(None, "9시에 시작해서 10시에 시작")
    assert merged.start_time is None
    merged = merge_slots(None, "9시에 시작해서 10시에 시작해서 11시에 끝나요")
    assert merged.start_time is None and merged.end_time == "11:00"


def test_conflict_with_previous_value_blanks_both_in_a_later_turn():
    # 실제 흐름에서는 저장된 시각이 slots.utterances에 말한 그대로 남아 있다 (이유는 그 발화를 다시 읽어 구한다)
    previous = full(start_time="13:00", end_time="15:00", utterances=["R-201 점검 3명 장비 없음 13시부터 15시까지"])
    merged = merge_slots(previous, "11시까지로 바꿔줘")
    assert merged.start_time is None and merged.end_time is None
    assert set(questions(merged).values()) == {time_reader.MSG_MISMATCH}


def test_answer_after_a_question_that_conflicts_with_saved_start():
    previous = full(start_time="13:00", utterances=["13시에 시작"])
    merged = merge_slots(previous.model_copy(update={"asked_keys": ["end_time"]}), "11시")
    assert merged.start_time is None and merged.end_time is None
    assert questions(merged)["start_time"] == time_reader.MSG_MISMATCH


def test_consistent_edit_keeps_working():
    previous = full(start_time="13:00", end_time="15:00")
    merged = merge_slots(previous, "종료 시간을 16시로 바꿔줘")
    assert (merged.start_time, merged.end_time) == ("13:00", "16:00")
    merged = merge_slots(previous, "시작을 10시로 바꿔줘")
    assert (merged.start_time, merged.end_time) == ("10:00", "15:00")
    merged = merge_slots(previous, "15시까지가 아니라 16시까지로")
    assert (merged.start_time, merged.end_time) == ("13:00", "16:00")


def test_reason_is_only_for_the_latest_message():
    first = merge_slots(None, "R-201 점검 9시 11시 14시")
    assert questions(first)["start_time"].startswith(time_reader.PREFIX_UNREAD)
    second = merge_slots(first, "3명이고 장비 없음")
    q = questions(second)
    assert not q["start_time"].startswith(time_reader.PREFIX_UNREAD) and q["start_time"] == time_reader.Q_BOTH
    third = merge_slots(first, "9시부터 11시까지")
    assert (third.start_time, third.end_time) == ("09:00", "11:00") and questions(third) == {}


# ── 문구: 모든 시간 되묻기에 '~부터 ~까지' 예시 ──

def test_wording_has_from_to_example():
    for text in (time_reader.Q_BOTH, time_reader.Q_START, time_reader.Q_END, time_reader.MSG_MISMATCH, time_reader.MSG_OVERNIGHT,
                 time_reader.REPLY_TIME_ROLE):
        assert "부터" in text and "까지" in text, text
    assert time_reader.PREFIX_UNREAD.strip() == "말씀하신 시간을 정확히 읽지 못했습니다."


def test_plain_questions_by_missing_part():
    assert questions(full()) == {"start_time": time_reader.Q_BOTH, "end_time": time_reader.Q_BOTH}
    assert questions(full(start_time="09:00")) == {"end_time": time_reader.Q_END}
    assert questions(full(end_time="11:00")) == {"start_time": time_reader.Q_START}
    assert questions(full(start_time="09:00", end_time="11:00")) == {}


def test_keys_and_count_are_unchanged_by_the_switch(monkeypatch):
    slots = full()
    on = [(f.key, f.label) for f in missing_fields(slots, ["일반"])]
    monkeypatch.setattr(time_reader, "TIME_READER_V2", False)
    off = [(f.key, f.label) for f in missing_fields(slots, ["일반"])]
    assert on == off == [("start_time", "시작 시간"), ("end_time", "종료 시간")]


def test_ask_message_merges_identical_time_questions_into_one_line():
    message = ask_message(missing_fields(full(), ["일반"]))
    assert message.count(time_reader.Q_BOTH) == 1 and "2." not in message
    both_gone = ask_message(missing_fields(WorkSlots(), ["일반"]))
    lines = [ln for ln in both_gone.splitlines()[1:]]
    assert len(lines) == len(set(lines)) and lines[0].startswith("1. ")


def test_question_after_asking_does_not_repeat_when_answered_with_label():
    first = merge_slots(None, "R-201 점검 3명 장비 없음")
    assert set(questions(first).values()) == {time_reader.Q_BOTH}
    answered = merge_slots(first.model_copy(update={"asked_keys": ["start_time", "end_time"]}), "시작은 9시, 종료는 11시")
    assert (answered.start_time, answered.end_time) == ("09:00", "11:00")
    half = merge_slots(first.model_copy(update={"asked_keys": ["start_time", "end_time"]}), "종료는 11시")
    assert (half.start_time, half.end_time) == (None, "11:00") and set(questions(half)) == {"start_time"}
    done = merge_slots(half.model_copy(update={"asked_keys": ["start_time"]}), "시작은 9시")
    assert (done.start_time, done.end_time) == ("09:00", "11:00")


# ── explain / conflict_kind ──

def test_conflict_kind_table():
    kind = time_reader.conflict_kind
    assert kind("09:00", "11:00") is None
    assert kind("09:00", "09:00") == "mismatch"
    assert kind("15:00", "11:00") == "mismatch"
    assert kind("17:59", "02:00") == "mismatch"  # 18시 전 시작은 야간으로 보지 않는다
    assert kind("18:00", "02:00") == "overnight"
    assert kind("23:00", "14:00") == "overnight"
    assert kind("19:00", "11:00", end_unmarked=True) == "mismatch"  # 11시에 12시간을 더하면 같은 날 (23시)
    assert kind("19:00", "11:00", end_unmarked=False) == "overnight"
    assert kind("19:00", "24:00") is None  # 밤 12시 종료
    assert kind(None, "11:00") is None and kind("09:00", None) is None


def test_explain_follows_the_latest_utterance_only():
    assert time_reader.explain([]) is None
    assert time_reader.explain(["9시 11시 14시"]) == "unread"
    assert time_reader.explain(["9시 11시 14시", "3명"]) is None
    assert time_reader.explain(["13시에 시작", "11시"]) == "mismatch"  # 이전 시작과 모순 (짧은 답)
    assert time_reader.explain(["13시에 시작", "11시까지"]) == "mismatch"
    assert time_reader.explain(["23시부터 2시까지"]) == "overnight"
    assert time_reader.explain(["9시부터 11시까지"]) is None


# ── 시간 말고 다른 항목은 읽기가 변하지 않는다 (스위치 on/off 비교) ──

OTHER_MESSAGES = [
    "오늘 13시에 화기작업 있어. 필요한 거 만들어줘.",
    "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 옆 T-105 탱크에 톨루엔. 3명, 15시까지. 크레인은 안 씀.",
    "산소 19%, LEL 3%, 화재감시자는 김철수, 3명, 장비 없음",
    "내일 오후 3시부터 5시까지 T-201 탱크 내부 청소",
    "3명이요",
]


@pytest.mark.parametrize("message", OTHER_MESSAGES)
def test_non_time_slots_are_identical_with_switch_off(monkeypatch, message):
    on = extract_slots(message, ["headcount", "lel"])
    monkeypatch.setattr(time_reader, "TIME_READER_V2", False)
    off = extract_slots(message, ["headcount", "lel"])
    drop = {"start_time", "end_time"}
    assert {k: v for k, v in on.items() if k not in drop} == {k: v for k, v in off.items() if k not in drop}


def test_switch_off_restores_the_old_misreads(monkeypatch):
    """끄면 변경 전 동작이다: 오독이 그대로 나온다 (스위치가 실제로 끄는지 확인)."""
    monkeypatch.setattr(time_reader, "TIME_READER_V2", False)
    assert times("11시에 마쳐") == ("11:00", None)  # 변경 전 오독
    assert times("15시까지가 아니라 16시까지로") == (None, "15:00")
    merged = merge_slots(None, "9시부터 9시까지")
    assert merged.end_time == "21:00"  # 12시간 보정
    assert questions(full())["start_time"] == "작업은 몇 시에 시작하나요?"
    assert slots_mod.ask_message(missing_fields(full(), ["일반"])).count("몇 시에") == 2


def test_ampm_ask_machinery_still_works_when_enabled(monkeypatch):
    """AMPM_ASK_HOURS를 켜는 현장 설정도 그대로 동작한다 (clarify 기존 장치)."""
    from app.agent import clarify

    monkeypatch.setattr(clarify, "AMPM_ASK_HOURS", frozenset(range(1, 12)))
    monkeypatch.setattr(clarify, "DAYTIME_PM_HOURS", frozenset())
    monkeypatch.setattr(clarify, "END_PM_HOURS", frozenset())
    merged = merge_slots(None, "R-201 점검 3명 장비 없음 3시부터 5시까지")
    assert merged.start_time is None  # 보류
    q = questions(merged)
    assert "오전인가요, 오후인가요" in q["start_time"]
    answered = merge_slots(merged, "오후")
    assert (answered.start_time, answered.end_time) == ("15:00", "17:00")
