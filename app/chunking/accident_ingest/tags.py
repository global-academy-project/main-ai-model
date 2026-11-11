"""태그 계산: 구조화 필드 → tag_codes 코드. 규칙은 프로필의 표에 있고, 여기서는 적용만 한다.

원칙: 불확실하면 비운다. 개요 키워드만으로는 판단하지 않는다(과소·오탐).
LLM 태깅은 하지 않는다. 서버가 태그 값을 검증하지 않으므로 validate 가 tag_codes 스냅샷과 대조한다.
"""

from __future__ import annotations

from typing import Any
from types import ModuleType

from . import config
from .config import BuildSettings


def _uniq(items: list[str]) -> list[str]:
    seen: list[str] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def accident_types(clean: dict[str, str], profile: ModuleType, settings: BuildSettings) -> list[str]:
    if clean["cateSeNm"] == "건설업":
        return list(profile.CONSTRUCTION_ACCIDENT_TYPE.get(clean["disasterType"], []))
    found = [code for code, rx in profile.FACTOR_RULES if rx.search(clean["disasterFactor"])]
    if not found and settings.overview_rescue:
        found = [code for code, rx in profile.FACTOR_RULES if rx.search(clean["disasterOverview"])]
    return _uniq(found)


def work_types(clean: dict[str, str], profile: ModuleType) -> list[str]:
    if clean["cateSeNm"] == "건설업":
        text = " ".join([clean["sifMclsfNm"], clean["sifSclsfNm"], clean["orgtNm"]])
        return _uniq([code for code, rx in profile.CONSTRUCTION_WORK_TYPE_RULES if rx.search(text)])
    code = profile.SITUATION_WORK_TYPE.get(profile.squash(clean["situation"]))
    return [code] if code else []


def hazards(clean: dict[str, str], accident: list[str], profile: ModuleType) -> list[str]:
    out: list[str] = []
    if "ELECTRIC" in accident:
        out.append("ELECTRICAL")
    chemical = "LEAK" in accident or profile.squash(clean["situation"]) == profile.HAZARD_CHEMICAL_SITUATION
    if clean["cateSeNm"] == "건설업" and clean["orgtNm"] == profile.HAZARD_CHEMICAL_ORGT:
        chemical = True
    if chemical:
        out.append("CHEMICAL")
    if "CAUGHT" in accident and "설비" in clean["disasterFactor"]:
        out.append("MECHANICAL")
    return _uniq(out)


def tags_for(doc: dict[str, Any], profile: ModuleType, settings: BuildSettings, year_max: int) -> dict[str, Any]:
    clean = doc["clean"]
    accident = accident_types(clean, profile, settings)
    return {
        "severity": profile.severity_of(clean["disasterOverview"]),
        "accident_type": accident,
        "hazard": hazards(clean, accident, profile),
        "work_type": work_types(clean, profile),
        "cas": [],
        "ghs": [],
        "year": profile.year_of(clean["disasterOverview"], config.YEAR_MIN, year_max),
    }
