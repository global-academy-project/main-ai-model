"""되묻기: 발화에서 필수 항목을 읽고(규칙), 빠진 항목을 체크리스트로 고른다.

필수 항목 (지침 2장 5번, 계획서 9장 시나리오 A의 되묻기 항목)
  - 작업 장소                       항상
  - 화기 종류                       화기작업일 때
  - 주변 취급 물질 ('없음'도 답)     화기·밀폐공간 작업일 때
  - 인원 · 종료 시간                 항상
  - 투입 장비 ('안 씀'·'없음'도 답)  항상
  - 작업 내용                       작업유형을 하나도 못 읽었을 때

새 발화에서 읽은 값만 덮어쓰고, 나머지는 이전 slots 값을 유지한다. 목록은 합친다.
LLM 판단은 2주차(계획서 8장). 지금은 text_rules의 키워드·정규식만 쓴다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.agent_work import MAX_UTTERANCES, MissingField, WorkSlots

from .text_rules import EQUIPMENT, MATERIALS, extract_locations, find_terms, is_negated, sentences, unique

_HOUR = r"(오전|오후)?\s*(\d{1,2})\s*시\s*(?:(\d{1,2})\s*분|(반))?"
_START_RE = re.compile(_HOUR + r"\s*(?:에|부터)")
_END_RE = re.compile(_HOUR + r"\s*(?:까지|종료|마감|끝)")
_END_TILDE_RE = re.compile(r"[~∼]\s*" + _HOUR)
_HEADCOUNT_RE = re.compile(r"(\d{1,3})\s*(?:명|인)(?!\s*분|치)")
_MATERIALS_NONE_RE = re.compile(r"(?:물질|가연물|위험물)\s*(?:은|는|이|도)?\s*(?:없|무)")
_EQUIPMENT_NONE_RE = re.compile(r"장비\s*(?:은|는|도)?\s*(?:없|안\s*씀|안\s*써|필요\s*없)")
_MEASUREMENT_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("O2", re.compile(r"(?:산소|O2)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*%")),
    ("LEL", re.compile(r"(?:LEL|가연성\s*가스)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*%")),
    ("H2S", re.compile(r"(?:황화수소|H2S)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*ppm", re.IGNORECASE)),
    ("CO", re.compile(r"(?:일산화탄소|CO(?!2))\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*ppm", re.IGNORECASE)),
    ("CO2", re.compile(r"(?:이산화탄소|탄산가스|CO2)\s*(?:농도)?\s*(?:는|은|가|이|:)?\s*(\d+(?:\.\d+)?)\s*%")),
)
_NOT_A_NAME = frozenset({"배치", "지정", "필요", "없음", "미정", "없이", "추가", "확인"})
_FIRE_WATCHER_RE = re.compile(r"화재\s*감시자\s*(?:는|은|:)?\s*([가-힣]{2,4})(?:씨|님)?(?=\s|$|[,.])")
_ATTENDANT_RE = re.compile(r"감시인\s*(?:는|은|:)?\s*([가-힣]{2,4})(?:씨|님)?(?=\s|$|[,.])")

# (화기 종류 표시, 찾는 표현, 같은 문장에 있으면 표시를 바꾸는 표현, 바뀐 표시)
_HOT_WORK_KINDS: tuple[tuple[str, tuple[str, ...], str | None, str | None], ...] = (
    ("가스 절단", ("가스절단", "산소절단", "가스 절단", "용단"), None, None),
    ("플라즈마 절단", ("플라즈마",), None, None),
    ("그라인더 연마", ("그라인더",), "절단", "그라인더 절단"),
    ("용접", ("용접",), None, None),
    ("토치 사용", ("토치",), "가열", "토치 가열"),
    ("납땜", ("납땜",), None, None),
)


def _to_hhmm(m: re.Match[str]) -> str | None:
    ampm, hour_s, minute_s, half = m.group(1), m.group(2), m.group(3), m.group(4)
    hour = int(hour_s)
    minute = 30 if half else int(minute_s or 0)
    if ampm == "오후" and hour < 12:
        hour += 12
    if hour > 24 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _hot_work_kind(text: str) -> str | None:
    kinds: list[str] = []
    for sentence in sentences(text):
        for label, keywords, modifier, modified in _HOT_WORK_KINDS:
            for kw in keywords:
                pos = sentence.find(kw)
                if pos == -1:
                    continue
                if is_negated(sentence, pos + len(kw)):
                    break
                kinds.append(modified if modifier and modifier in sentence else label)
                break
    # '그라인더 절단'이 있으면 같은 공구의 다른 표시는 뺀다
    result = unique(kinds)
    if "그라인더 절단" in result and "그라인더 연마" in result:
        result.remove("그라인더 연마")
    return ", ".join(result) or None


def extract_slots(message: str) -> dict:
    """발화 하나에서 읽은 값만 돌려준다 (못 읽은 항목은 키가 없다)."""
    found: dict = {}
    location, material_locations = extract_locations(message)
    if location:
        found["location"] = location
    if material_locations:
        found["material_locations"] = material_locations
    if kind := _hot_work_kind(message):
        found["hot_work_kind"] = kind

    materials, _ = find_terms(message, MATERIALS)
    if materials:
        found["nearby_materials"] = materials
    if _MATERIALS_NONE_RE.search(message):
        found["nearby_materials_none"] = True

    equipment, excluded = find_terms(message, EQUIPMENT)
    if equipment:
        found["equipment"] = equipment
    if excluded:
        found["excluded_equipment"] = excluded
    if _EQUIPMENT_NONE_RE.search(message):
        found["equipment_none"] = True

    if m := _HEADCOUNT_RE.search(message):
        found["headcount"] = int(m.group(1))
    if (m := _START_RE.search(message)) and (hhmm := _to_hhmm(m)):
        found["start_time"] = hhmm
    end_match = _END_RE.search(message) or _END_TILDE_RE.search(message)
    if end_match and (hhmm := _to_hhmm(end_match)):
        found["end_time"] = hhmm

    measurements = {item: float(m.group(1)) for item, pattern in _MEASUREMENT_RES if (m := pattern.search(message))}
    if measurements:
        found["measurements"] = measurements
    if (m := _FIRE_WATCHER_RE.search(message)) and m.group(1) not in _NOT_A_NAME:
        found["fire_watcher"] = m.group(1)
    if (m := _ATTENDANT_RE.search(message)) and m.group(1) not in _NOT_A_NAME:
        found["attendant"] = m.group(1)
    return found


def merge_slots(previous: WorkSlots | None, message: str) -> WorkSlots:
    base = (previous or WorkSlots()).model_dump()
    base["utterances"] = (base["utterances"] + [message.strip()])[-MAX_UTTERANCES:]
    found = extract_slots(message)
    for key, value in found.items():
        if isinstance(value, list):
            base[key] = unique([*base.get(key, []), *value])
        elif isinstance(value, dict):
            base[key] = {**base.get(key, {}), **value}
        else:
            base[key] = value
    # 나중에 '안 씀'이라고 한 장비는 사용 목록에서 뺀다 (반대도 같다)
    latest_excluded = found.get("excluded_equipment", [])
    latest_used = found.get("equipment", [])
    base["equipment"] = [e for e in base["equipment"] if e not in latest_excluded]
    base["excluded_equipment"] = [e for e in base["excluded_equipment"] if e not in latest_used]
    if base["nearby_materials"]:
        base["nearby_materials_none"] = False
    return WorkSlots.model_validate(base)


def full_text(slots: WorkSlots) -> str:
    return "\n".join(slots.utterances)


_QUESTIONS: dict[str, tuple[str, str]] = {
    "task": ("작업 내용", "어떤 작업인지 대상 설비와 작업 내용을 알려주세요."),
    "location": ("작업 장소", "작업 장소(설비 번호나 구역)는 어디인가요?"),
    "hot_work_kind": ("화기 종류", "화기 종류는 무엇인가요? (용접 · 절단 · 그라인더 · 토치 등)"),
    "nearby_materials": ("주변 취급 물질", "작업 장소 주변에 취급하거나 보관 중인 물질이 있나요? 없으면 '물질 없음'이라고 알려주세요."),
    "headcount": ("작업 인원", "작업 인원은 몇 명인가요?"),
    "end_time": ("종료 시간", "작업은 몇 시에 끝나나요?"),
    "equipment": ("투입 장비", "투입하는 장비·공구를 알려주세요. 쓰지 않는 장비는 '크레인 안 씀'처럼 알려주셔도 됩니다."),
}


def missing_fields(slots: WorkSlots, work_types: Sequence[str]) -> list[MissingField]:
    keys: list[str] = []
    if not work_types:
        keys.append("task")
    if not slots.location:
        keys.append("location")
    if "화기" in work_types and not slots.hot_work_kind:
        keys.append("hot_work_kind")
    if ("화기" in work_types or "밀폐공간" in work_types) and not (slots.nearby_materials or slots.nearby_materials_none):
        keys.append("nearby_materials")
    if slots.headcount is None:
        keys.append("headcount")
    if slots.end_time is None:
        keys.append("end_time")
    if not (slots.equipment or slots.excluded_equipment or slots.equipment_none):
        keys.append("equipment")
    return [MissingField(key=k, label=_QUESTIONS[k][0], question=_QUESTIONS[k][1]) for k in keys]


def ask_message(missing: Sequence[MissingField]) -> str:
    lines = ["초안을 만들려면 몇 가지가 더 필요합니다."]
    lines += [f"{i}. {f.question}" for i, f in enumerate(missing, 1)]
    return "\n".join(lines)
