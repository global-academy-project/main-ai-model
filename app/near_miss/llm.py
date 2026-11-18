from __future__ import annotations

import json
import re

from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.rag.tags.metadata import is_valid_cas_number

from .schemas import NearMissTagAxis, NearMissTagSuggestion
from .tagging import TAG_CODES

SYSTEM_PROMPT = """아차사고를 5축 분류 코드로 제안합니다.
응답은 JSON 객체 하나만 출력합니다. 규칙으로 이미 나온 축은 덮어쓰지 않습니다.
형식: {"accidentType":["FALL"],"hazard":["HUMAN"],"workType":["HEIGHT"],"substance":["108-88-3"]}
허용 코드만 사용하고, 확실하지 않은 축은 빈 배열로 반환합니다."""


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

    prompt = json.dumps(
        {
            "text": request_text,
            "missing_axes": missing_axes,
            "allowed_codes": {
                "accidentType": TAG_CODES["ACCIDENT_TYPE"],
                "hazard": TAG_CODES["HAZARD"],
                "workType": TAG_CODES["WORK_TYPE"],
                "substance": "valid CAS-xxxx number",
            },
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

    if "SUBSTANCE" in missing_axes:
        values = parsed.get("substance")
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and is_valid_cas_number(value):
                    result.append(NearMissTagSuggestion(axis="SUBSTANCE", value=value, confidence=0.5))

    suggested = len(result) > len(existing)
    return result, (response.model or None) if suggested else None
