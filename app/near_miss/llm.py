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
- 축마다 문장 근거가 가장 강한 코드 하나만 고르고, 판단 근거가 없으면 빈 배열([])로 답합니다.
- 허용 코드표의 코드만 사용합니다.
- 규칙으로 이미 나온 축은 덮어쓰지 않습니다.
- 작업 유형(workType)은 보고에서 실제로 수행했거나 준비·시도한 작업을 문장 근거로 판단합니다.
- 아래 작업 유형 판정은 직접 근거가 있을 때만 사용합니다.
  - HOT: 용접, 용단, 절단, 그라인더, 토치 사용, 불티·불꽃을 동반한 열 작업을 직접 수행하거나 준비했다.
  - CONFINED: 맨홀, 탱크, 사일로, 피트, 집수정, 저장조, 반응기 등의 내부에 진입했거나 진입하려 했다.
  - HEIGHT: 사다리, 비계, 고소작업대, 지붕 위 또는 높이 작업발판에서 작업했거나 그 위에서 작업을 시도했다.
  - ELECTRICAL: 분전반, 배전반, 차단기, 전선, 콘센트, 충전부를 점검·조작·교체 또는 수리했다.
  - LIFTING: 지게차, 크레인, 호이스트, 체인블록 등 양중 설비를 사용했거나 중량물·팔레트를 들어 올리거나 운반했다.
- 위 작업이 실제로 확인되지 않으면 GENERAL을 선택합니다.
- 장비명이나 장소명이 위험을 암시하더라도 실제 작업행위가 없으면 해당 작업 유형으로 선택하지 않습니다.
- 탱크·반응기·맨홀은 외부 점검만 했거나 "외부", "주변", "내부 진입 없음"이 명시되면 CONFINED가 아니라 GENERAL 또는 실제 외부 작업 유형입니다.
- 용접·절단·그라인더·토치·불티·불꽃 표현이 없으면 HOT를 선택하지 않습니다.
- 분전반 옆 작업만 있고 전기 설비를 점검·조작하지 않았다면 ELECTRICAL이 아니라 HOT 또는 GENERAL을 판단합니다.
- 사다리·비계 근처의 낙하물이나 주변 통제만 있고 그 위에서 작업하지 않았다면 HEIGHT만으로 강제 분류하지 않습니다.
- 묵시적 추측보다 직접 근거를 우선합니다. 문장에 여러 작업이 명시되면 해당 축에서 가장 강한 직접 근거 하나를 선택합니다.
- 보호구·안전대·절차·감시자 미준수가 원인이면 인적·관리(HUMAN)를 우선합니다.
- 산소 부족·유해가스·고온·조도·바닥·난간 상태가 원인이면 작업환경(ENVIRONMENTAL)을 우선합니다.
- 설비·회전체·중량물·낙하물이 원인이면 기계적(MECHANICAL)을 우선합니다.

예시
보고: "용접 중 불티가 옆 신나 통으로 튀었다"
답: {"hazard":["CHEMICAL"],"workType":["HOT"]}
보고: "콘센트를 교체하는 작업 중 회로 차단기의 불안정함을 확인했다"
답: {"hazard":["ELECTRICAL"],"workType":["ELECTRICAL"]}
보고: "배수로 덮개가 열려 있어 걸려 넘어질 뻔했다"
답: {"hazard":["ENVIRONMENTAL"],"workType":["GENERAL"]}
보고: "작업자가 사다리를 이용해 천장 조명 교체 중 추락할 뻔했다"
답: {"accidentType":["FALL"],"hazard":["ENVIRONMENTAL"],"workType":["HEIGHT"]}
보고: "위험 구역에 보호구를 착용하지 않고 들어가 제지당했다"
답: {"hazard":["HUMAN"],"workType":["GENERAL"]}
보고: "반응기 외부에서 용접 작업을 했으며 내부 진입은 하지 않았다"
답: {"hazard":["CHEMICAL"],"workType":["HOT"]}
보고: "탱크 외부 맨홀 주변에서 누출 상태를 점검했지만 내부에는 진입하지 않았다"
답: {"accidentType":["LEAK"],"hazard":["CHEMICAL"],"workType":["GENERAL"]}
보고: "지게차로 팔레트를 들어 올릴 때 적재물이 흔들려 주변을 피했다"
답: {"hazard":["MECHANICAL"],"workType":["LIFTING"]}"""


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
