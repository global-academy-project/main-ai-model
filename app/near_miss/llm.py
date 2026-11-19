from __future__ import annotations

import json
import re

from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.rag.tags.metadata import is_valid_cas_number

from .schemas import NearMissTagAxis, NearMissTagSuggestion
from .tagging import TAG_CODES

SYSTEM_PROMPT = """현장 아차사고 보고를 분류 코드로 제안합니다.
응답은 JSON 객체 하나만 출력합니다.
- 질문의 missing_axes에 있는 축만 답합니다.
- 축마다 가장 적절한 코드 하나만 고르고, 판단 근거가 없으면 빈 배열([])로 답합니다.
- 허용 코드표의 코드만 사용합니다.
- 규칙으로 이미 나온 축은 덮어쓰지 않습니다.

예시
보고: "용접 중 불티가 옆 신나 통으로 튀었다"
답: {"hazard":["CHEMICAL"],"workType":["HOT"]}
보고: "배수로 덮개가 열려 있어 걸려 넘어질 뻔했다"
답: {"hazard":["ENVIRONMENTAL"],"workType":["GENERAL"]}"""


CODE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "ACCIDENT_TYPE": {
        "FIRE": "화재",
        "EXPLOSION": "폭발",
        "LEAK": "누출",
        "ASPHYXIATION": "질식",
        "FALL": "떨어짐",
        "TRIP": "넘어짐",
        "CAUGHT": "끼임",
        "STRUCK": "맞음",
        "COLLAPSE": "붕괴",
        "ELECTRIC": "감전",
    },
    "HAZARD": {
        "MECHANICAL": "기계적 — 설비·회전체·차량·떨어지는 물체 등 기계적 힘",
        "ELECTRICAL": "전기적 — 분전반·차단기·전선·충전부 등 전기",
        "CHEMICAL": "화학적 — 화학물질·가스 누출, 인화물 접촉 또는 불꽃이 인화물에 닿음",
        "ENVIRONMENTAL": "작업환경 — 산소 부족, 유해가스, 고온, 조명·바닥 등 장소 조건",
        "HUMAN": "인적·관리 — 보호구 미착용, 절차 미준수, 감시자 없음 등 사람·관리 문제",
    },
    "WORK_TYPE": {
        "GENERAL": "일반 — 아래 작업 유형에 해당하지 않음",
        "HOT": "화기 — 용접·절단·그라인더 등 불꽃 또는 열을 사용하는 작업",
        "CONFINED": "밀폐공간 — 맨홀·탱크·피트·사일로 내부 진입",
        "HEIGHT": "고소 — 사다리·비계·고소작업대 또는 2m 이상 작업",
        "ELECTRICAL": "전기 — 전기설비 점검 또는 조작",
        "LIFTING": "중량물·양중 — 지게차·크레인·호이스트·중량물 운반",
    },
}


def suggest_missing_tags(
    llm: LLMClient,
    *,
    request_text: str,
    existing: list[NearMissTagSuggestion],
) -> tuple[list[NearMissTagSuggestion], str | None]:
    existing_axes = {tag.axis for tag in existing}
    missing_axes = [
        axis for axis in ("ACCIDENT_TYPE", "HAZARD", "WORK_TYPE", "SUBSTANCE")
        if axis not in existing_axes
    ]
    if not missing_axes:
        return existing, None

    allowed_codes: dict[str, object] = {}
    key_by_axis = {
        "ACCIDENT_TYPE": "accidentType",
        "HAZARD": "hazard",
        "WORK_TYPE": "workType",
        "SUBSTANCE": "substance",
    }
    for axis in missing_axes:
        key = key_by_axis[axis]
        if axis == "SUBSTANCE":
            allowed_codes[key] = "유효한 CAS 번호 (예: 108-88-3)"
        else:
            allowed_codes[key] = {
                code: CODE_DESCRIPTIONS[axis][code]
                for code in TAG_CODES[axis]
                if code in CODE_DESCRIPTIONS[axis]
            }

    prompt = json.dumps(
        {
            "text": request_text,
            "missing_axes": missing_axes,
            "allowed_codes": allowed_codes,
        },
        ensure_ascii=False,
    )
    response = llm.chat([Message(role="user", content=prompt)], system=SYSTEM_PROMPT, max_tokens=300)

    matched = re.search(r"\{.*\}", response.text, re.DOTALL)
    if matched is None:
        return existing, None
    try:
        parsed = json.loads(matched.group(0))
    except json.JSONDecodeError:
        return existing, None

    result = list(existing)
    mappings: tuple[tuple[str, NearMissTagAxis, list[str]], ...] = (
        ("accidentType", "ACCIDENT_TYPE", TAG_CODES["ACCIDENT_TYPE"]),
        ("hazard", "HAZARD", TAG_CODES["HAZARD"]),
        ("workType", "WORK_TYPE", TAG_CODES["WORK_TYPE"]),
    )
    for key, axis, allowed in mappings:
        if axis not in missing_axes:
            continue
        values = parsed.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value in allowed:
                result.append(NearMissTagSuggestion(axis=axis, value=value, confidence=0.5))
                break

    if "SUBSTANCE" in missing_axes:
        values = parsed.get("substance")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and is_valid_cas_number(value):
                    result.append(NearMissTagSuggestion(axis="SUBSTANCE", value=value, confidence=0.5))

    suggested = len(result) > len(existing)
    return result, (response.model or None) if suggested else None
