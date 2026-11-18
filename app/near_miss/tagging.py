from __future__ import annotations

import json
import re
from pathlib import Path

from app.chunking.accident_ingest.profiles.sif import FACTOR_RULES, CONSTRUCTION_WORK_TYPE_RULES
from app.rag.tags.metadata import is_valid_cas_number

from .schemas import NearMissTagSuggestion

TAGGING_VERSION = "nm-1"
TAG_CODES = json.loads(
    (Path(__file__).parents[1] / "chunking" / "accident_ingest" / "profiles" / "tag_codes_v5.json")
    .read_text(encoding="utf-8")
)

CAS_PATTERN = re.compile(r"[0-9]{2,7}-[0-9]{2}-[0-9]")


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

    accident_types = list(dict.fromkeys(code for code, pattern in FACTOR_RULES if pattern.search(text)))
    for code in accident_types:
        if code in TAG_CODES["ACCIDENT_TYPE"]:
            tags.append(NearMissTagSuggestion(axis="ACCIDENT_TYPE", value=code, confidence=0.8))

    hazards: list[str] = []
    if "ELECTRIC" in accident_types:
        hazards.append("ELECTRICAL")
    if "LEAK" in accident_types:
        hazards.append("CHEMICAL")
    if "STRUCK" in accident_types or "CAUGHT" in accident_types:
        hazards.append("MECHANICAL")
    for code in dict.fromkeys(hazards):
        if code in TAG_CODES["HAZARD"]:
            tags.append(NearMissTagSuggestion(axis="HAZARD", value=code, confidence=0.8))

    suggested_work_types: list[str] = []
    if work_type and work_type in TAG_CODES["WORK_TYPE"] and work_type != "GENERAL":
        suggested_work_types.append(work_type)
    else:
        suggested_work_types.extend(
            code for code, pattern in CONSTRUCTION_WORK_TYPE_RULES if pattern.search(text)
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
