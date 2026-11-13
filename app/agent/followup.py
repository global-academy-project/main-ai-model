"""DONE 뒤의 다음 말을 가르는 규칙 (11/13, 후속 말 처리).

문제: DONE(초안 완료) 뒤에 온 말도 파이프라인 전체(계획 LLM 22~35초 포함)를 다시 돌렸다. "고마워"에도 초안을 다시 만들고,
"그라인더 말고 용접으로"는 두 값이 합쳐지고, 다른 작업을 말해도 이전 slots가 섞였다.

DONE 뒤의 말을 코드로 세 가지로 가른다 (LLM 없음, 응답·slots 스키마 변경 없음). 상태는 slots에서 읽는다:
이전 slots의 필수 항목이 모두 차 있고(되묻기가 끝남) asked_keys가 비어 있으면 DONE 뒤다.

  무변경   읽은 값이 하나도 안 바뀜 ("고마워", "위험요인 알려줘")  → 파이프라인을 돌리지 않고 안내만 한다. slots는 그대로.
  수정     값이 바뀜 (인원·시각·날짜·장비·감시자·측정값, "그라인더 말고 용접") → 이전 slots에 병합해 다시 만들고,
           ask.message 앞에 "변경: 인원 3명 → 4명"을 붙인다 (코드가 만든다).
  새 작업  작업 카테고리가 바뀜(이전에 없던 화기·밀폐공간·고소·전기·중량물이 나옴, 장비 이름만으로는 안 봄), 또는 장소가
           다르고 수정 표지어("바꿔·말고·대신…")가 없고 작업 내용이 있음 → NEW_TASK_MODE가 정한다.
             "guide"  (기본, 사용자 결정 11/13) 처리하지 않고 안내만 한다: 새 작업은 '새 대화'를 눌러 진행하도록.
                      slots는 이전 그대로 돌려주므로 기존 초안이 사라지지 않는다 (오판해도 잃는 것이 없다).
             "reset"  (선택) 이전 slots를 버리고 이번 말을 새 작업의 첫 말로 처리한다. 같은 대화에서 DONE이 다시 오면
                      Spring이 앞선 초안을 덮어쓴다 (계획서 7-2장). 오판하면 이전 slots를 잃는다.
  날짜만 바뀐 말은 수정이다 (사용자 결정 11/13). "다시 만들어줘"는 값이 안 바뀌어도 수정(재생성)이다.

DONE 뒤가 아니면(첫 말, 되묻기 중) judge_followup은 None을 돌려주고 기존 흐름이 그대로 돈다.
끄기: FOLLOWUP_ENABLED = False → 항상 None (변경 전과 같다). 새 작업 처리 방식은 NEW_TASK_MODE.
교체 표현("그라인더 말고 용접")은 text_rules.REPLACE_NEGATION이 정한다 (부정 표현에 말고·대신·아니라·아니고를 더함).

키워드는 임시값이다 (정답 세트 전).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.schemas.agent_work import WorkSlots

from . import time_reader
from .classify import classify_text
from .slots import full_text, merge_slots, missing_fields
from .text_rules import EQUIPMENT, extract_locations

FOLLOWUP_ENABLED = True
NEW_TASK_MODE: Literal["reset", "guide"] = "guide"

# 새 작업 판단에 쓰는 카테고리. '일반'은 작업 내용이 있다는 신호일 뿐이라 카테고리로 보지 않는다.
HIGH_RISK_CATEGORIES = frozenset({"화기", "밀폐공간", "고소", "전기", "중량물"})
# 장비 이름이 카테고리를 만들면(크레인→중량물, 사다리→고소, 그라인더→화기) 부가 정보 수정으로 본다.
_EQUIPMENT_WORDS = frozenset(w for aliases in EQUIPMENT.values() for w in aliases) | frozenset(EQUIPMENT)

_EDIT_RE = re.compile(r"바꿔|바꾸|바꿀|바꿨|변경|수정|정정|고쳐|말고(?!도)|대신|아니라|아니고|[으]?로\s*할게")
_REGENERATE_RE = re.compile(r"다시\s*(?:만들|작성|생성)|재생성|재작성|새로\s*만들")
_TAG_RE = re.compile(r"^[A-Z]{1,4}-?\d{1,4}[A-Z]?$")

NOTE_NEW_TASK = "이전 작업과 다른 작업으로 보고 새 작업으로 시작합니다.\n"
NOTE_REGENERATED = "변경 없이 다시 만들었습니다.\n"
REPLY_UNCHANGED = (
    "초안은 그대로입니다. 고칠 내용이 있으면 '인원 4명으로', '15시까지'처럼 말씀해 주세요. "
    "다른 작업을 등록하려면 '새 대화'를 눌러 진행해 주세요."
)
REPLY_WHAT_TO_CHANGE = "무엇을 어떻게 바꿀지 알려주세요. 예: '인원 4명으로', '15시까지', '화재감시자는 김철수'."
REPLY_NEW_TASK = (
    "이전 작업과 다른 작업으로 보입니다. 새 작업은 '새 대화'를 눌러 진행해 주세요. 지금 초안은 그대로 남아 있습니다. "
    "지금 작업을 고치려는 거라면 바꿀 값만 말씀해 주세요 (예: '인원 4명으로')."
)

# (WorkSlots 필드, 변경 요약에 쓰는 이름)
_FIELDS: tuple[tuple[str, str], ...] = (
    ("location", "장소"),
    ("hot_work_kind", "화기 종류"),
    ("nearby_materials", "주변 물질"),
    ("nearby_materials_none", "주변 물질 없음"),
    ("material_locations", "물질 위치"),
    ("headcount", "인원"),
    ("work_date", "날짜"),
    ("start_time", "시작"),
    ("end_time", "종료"),
    ("equipment", "장비"),
    ("excluded_equipment", "제외 장비"),
    ("equipment_none", "장비 없음"),
    ("measurements", "측정값"),
    ("fire_watcher", "화재감시자"),
    ("attendant", "감시인"),
)


@dataclass(frozen=True)
class Followup:
    """DONE 뒤의 말을 어떻게 처리할지.

    kind    "none" | "edit" | "new"
    slots   이번 요청에서 쓸 slots (none·guide 안내는 이전 그대로, edit은 병합 결과, new(reset)는 새 slots)
    reply   있으면 파이프라인을 돌리지 않고 이 문장으로 WAITING_INPUT 안내를 돌려준다
    note    파이프라인 응답(ask.message) 앞에 붙일 한 줄 (끝에 줄바꿈 포함)
    """

    kind: Literal["none", "edit", "new"]
    slots: WorkSlots
    reply: str | None = None
    note: str = ""


def message_categories(text: str) -> set[str]:
    """말 하나에서 읽은 고위험 카테고리. 장비 이름만으로 생긴 카테고리는 뺀다."""
    keywords: dict[str, list[str]] = {}
    for reason in classify_text(text).reasons:
        keywords.setdefault(reason.work_type, []).append(reason.keyword)
    return {w for w, kws in keywords.items() if w in HIGH_RISK_CATEGORIES and not all(k in _EQUIPMENT_WORDS for k in kws)}


def _previous_categories(previous: WorkSlots) -> set[str]:
    return set(classify_text(full_text(previous)).work_types) & HIGH_RISK_CATEGORIES


def is_done_state(previous: WorkSlots | None) -> bool:
    """이전 slots가 DONE 뒤의 상태인가: 작업 등록 상태가 있고, 되묻는 중이 아니고, 필수 항목이 모두 차 있다."""
    if previous is None or not previous.utterances or previous.asked_keys or previous.pending_text:
        return False
    work_types = classify_text(full_text(previous)).work_types
    return not missing_fields(previous, work_types)


def _tags(location: str) -> set[str]:
    return {t for t in location.split() if _TAG_RE.match(t)}


def _same_location(a: str, b: str) -> bool:
    tags_a, tags_b = _tags(a), _tags(b)
    if tags_a and tags_b:
        return bool(tags_a & tags_b)  # 설비 번호가 있으면 번호로 본다 (R-201 ≠ R-305, 나머지 단어가 같아도)
    ca, cb = "".join(a.split()), "".join(b.split())
    return ca in cb or cb in ca


def is_new_task(previous: WorkSlots, message: str) -> bool:
    cats = message_categories(message)
    if cats and not cats <= _previous_categories(previous):
        return True  # 카테고리가 바뀜 (장소가 같아도)
    location, _ = extract_locations(message)
    if (
        location
        and previous.location
        and not _same_location(location, previous.location)
        and not _EDIT_RE.search(message)
        and classify_text(message).work_types  # 작업 내용이 있는 말 ('T-105 탱크 옆에 메탄올 있어'는 아님)
    ):
        return True
    return False


def _fmt(field: str, value: object) -> str:
    if value in (None, "", [], {}):
        return "없음"
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if field == "headcount":
        return f"{value}명"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    return str(value)


def changes(previous: WorkSlots, merged: WorkSlots) -> list[str]:
    """바뀐 항목 '이름 이전 → 이후'. 발화·되묻기 기록은 비교하지 않는다."""
    out: list[str] = []
    for field, label in _FIELDS:
        old, new = getattr(previous, field), getattr(merged, field)
        if old != new:
            out.append(f"{label} {_fmt(field, old)} → {_fmt(field, new)}")
    return out


def judge_followup(previous: WorkSlots | None, message: str, *, today: date | None = None) -> Followup | None:
    """DONE 뒤의 말이면 처리 방식을, 아니면 None (기존 흐름)."""
    if not FOLLOWUP_ENABLED or not is_done_state(previous):
        return None
    assert previous is not None  # is_done_state가 확인했다
    marker = _EDIT_RE.search(message) is not None

    if is_new_task(previous, message):
        if NEW_TASK_MODE == "guide":
            return Followup("new", previous, reply=REPLY_NEW_TASK)
        return Followup("new", merge_slots(None, message, today=today), note=NOTE_NEW_TASK)

    merged = merge_slots(previous, message, today=today)
    diff = changes(previous, merged)
    if diff:
        return Followup("edit", merged, note=f"변경: {', '.join(diff)}\n")
    if _REGENERATE_RE.search(message):
        return Followup("edit", previous, note=NOTE_REGENERATED)
    if time_reader.TIME_READER_V2 and time_reader.has_unread_time(message):  # '16시로 변경'처럼 시작·종료를 정할 수 없는 시각
        return Followup("none", previous, reply=time_reader.REPLY_TIME_ROLE)
    return Followup("none", previous, reply=REPLY_WHAT_TO_CHANGE if marker else REPLY_UNCHANGED)


__all__ = ["FOLLOWUP_ENABLED", "NEW_TASK_MODE", "Followup", "judge_followup", "is_done_state", "is_new_task"]
