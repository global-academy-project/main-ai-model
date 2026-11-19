from __future__ import annotations

import json
import re
from pathlib import Path

from app.chunking.accident_ingest.profiles.sif import FACTOR_RULES, CONSTRUCTION_WORK_TYPE_RULES
from app.rag.tags.metadata import is_valid_cas_number

from .schemas import NearMissTagSuggestion

TAGGING_VERSION = "nm-3"
TAG_CODES = json.loads(
    (Path(__file__).parents[1] / "chunking" / "accident_ingest" / "profiles" / "tag_codes_v5.json")
    .read_text(encoding="utf-8")
)

CAS_PATTERN = re.compile(r"[0-9]{2,7}-[0-9]{2}-[0-9]")

# 아차사고 현장 표현 전용 보조 규칙.
# 공공 사고자료의 FACTOR_RULES와 CONSTRUCTION_WORK_TYPE_RULES는 바꾸지 않는다.
NEAR_MISS_ACCIDENT_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FIRE", re.compile(r"화재|불티|불꽃|산소절단|토치")),
    ("EXPLOSION", re.compile(r"폭발|파열|가연성 먼지|압력 상승")),
    ("LEAK", re.compile(r"누출|샌다|흘렀|새어|약품 유출")),
    ("ASPHYXIATION", re.compile(r"질식|산소 부족|저산소|숨이 막|숨을 못|어지러워")),
    ("FALL", re.compile(r"추락|떨어질 뻔|떨어질 듯|헛디딤|헛디뎌|균형을 잃")),
    ("TRIP", re.compile(r"넘어질 뻔|넘어질 듯|걸려 넘어|발이 걸려")),
    ("CAUGHT", re.compile(r"끼일 뻔|말려 들어|말려들어|회전체|프레스")),
    ("STRUCK", re.compile(r"낙하물|자재가 떨어|공구가 떨어|맞을 뻔|비래|충돌")),
    ("COLLAPSE", re.compile(r"무너질 뻔|붕괴|기울어|지지대")),
    ("ELECTRIC", re.compile(r"감전|찌릿|누전|분전반|차단기|충전부")),
)

NEAR_MISS_WORK_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("HOT", re.compile(r"용접|용단|그라인더|절단|불티|불꽃|토치|산소절단")),
    ("CONFINED", re.compile(r"밀폐공간|맨홀|저장조|집수정|반응기 내부|사일로|피트 안|탱크 내부")),
    ("HEIGHT", re.compile(r"고소작업대|사다리|비계|난간|지붕 위|천장|작업발판|[2-9]\s*m|[2-9]\s*미터")),
    ("ELECTRICAL", re.compile(r"분전반|배전반|차단기|전선|콘센트|누전|충전부|활선")),
    ("LIFTING", re.compile(r"지게차|호이스트|체인블록|크레인|중량물|파레트|팔레트|줄걸이")),
)


def suggest_near_miss_tags(
    *,
    title: str,
    description: str,
    immediate_action: str | None,
    work_type: str | None,
) -> list[NearMissTagSuggestion]:
    text = "\n".join(value for value in (title, description, immediate_action) if value)

    tags = [
        NearMissTagSuggestion(axis="SEVERITY", value="NEAR_MISS", confidence=1.0),
    ]

    accident_types = list(dict.fromkeys(
        code
        for code, pattern in (*FACTOR_RULES, *NEAR_MISS_ACCIDENT_TYPE_RULES)
        if pattern.search(text)
    ))
    for code in accident_types:
        if code in TAG_CODES["ACCIDENT_TYPE"]:
            tags.append(NearMissTagSuggestion(axis="ACCIDENT_TYPE", value=code, confidence=0.8))

    hazards: list[str] = []
    if "ELECTRIC" in accident_types:
        hazards.append("ELECTRICAL")
    if "FIRE" in accident_types or "EXPLOSION" in accident_types or "LEAK" in accident_types:
        hazards.append("CHEMICAL")
    if "ASPHYXIATION" in accident_types:
        hazards.append("ENVIRONMENTAL")
    if "FALL" in accident_types or "TRIP" in accident_types:
        if re.search(r"보호구|안전대|안전화|절차|교육|감시자|훈련", text):
            hazards.append("HUMAN")
        elif re.search(r"장비|사다리|비계|발판|난간|기둥|지지대", text):
            hazards.append("MECHANICAL")
        else:
            hazards.append("ENVIRONMENTAL")
    if "COLLAPSE" in accident_types:
        hazards.append("MECHANICAL")
    if "STRUCK" in accident_types or "CAUGHT" in accident_types:
        hazards.append("MECHANICAL")
    for code in dict.fromkeys(hazards):
        if code in TAG_CODES["HAZARD"]:
            tags.append(NearMissTagSuggestion(axis="HAZARD", value=code, confidence=0.8))

    suggested_work_types: list[str] = []
    if work_type and work_type in TAG_CODES["WORK_TYPE"] and work_type != "GENERAL":
        suggested_work_types.append(work_type)
    else:
        confined_entry_text = re.search(
            r"(?:내부|안쪽|입구|진입|들어가|오르내리)",
            text,
        )
        external_only_text = re.search(
            r"외부|밖에서|내부에는\s*진입하지|내부로는\s*진입하지",
            text,
        )
        for code, pattern in CONSTRUCTION_WORK_TYPE_RULES:
            if code == "CONFINED" and (external_only_text or not confined_entry_text):
                continue
            if pattern.search(text):
                suggested_work_types.append(code)
        suggested_work_types.extend(
            code for code, pattern in NEAR_MISS_WORK_TYPE_RULES if pattern.search(text)
        )
    for code in dict.fromkeys(suggested_work_types):
        if code in TAG_CODES["WORK_TYPE"]:
            tags.append(
                NearMissTagSuggestion(
                    axis="WORK_TYPE",
                    value=code,
                    confidence=0.9 if work_type else 0.8,
                )
            )

    for match in CAS_PATTERN.findall(text):
        if is_valid_cas_number(match):
            tags.append(NearMissTagSuggestion(axis="SUBSTANCE", value=match, confidence=1.0))

    return tags
