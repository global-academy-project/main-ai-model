"""fill_form 도구 (B, 실제 구현): 관리 항목 초안을 항목 JSON 스키마(app/schemas/forms.py)로 검증한다.

입력  {"form_type": "RISK_ASSESSMENT" | "HOT_WORK_PERMIT" | ..., "data": {...}}
출력  {"form_type": ..., "data": {...검증·정규화된 값, 계산 필드(risk_score 등) 포함}}
실패  ToolError("FORM_INVALID") — 검증 오류 원문은 로그에만 남긴다 (메시지는 고정 문구, 지침 4장).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from pydantic import ValidationError
from app.schemas.forms import FORM_MODELS
from app.exceptions import ToolError

logger = logging.getLogger(__name__)

FILL_FORM_PARAMETERS = {
    "type": "object",
    "properties": {
        "form_type": {"type": "string", "enum": sorted(FORM_MODELS)},
        "data": {"type": "object"},
    },
    "required": ["form_type", "data"],
}


def fill_form(arguments: Mapping[str, Any]) -> dict[str, Any]:
    form_type, data = arguments.get("form_type"), arguments.get("data")
    model = FORM_MODELS.get(form_type) if isinstance(form_type, str) else None
    if model is None:
        raise ToolError("FORM_TYPE_UNKNOWN", "지원하지 않는 양식 종류입니다.", tool="fill_form")
    if not isinstance(data, Mapping):
        raise ToolError("FORM_INVALID", "양식 데이터(data)가 객체가 아닙니다.", tool="fill_form")
    try:
        validated = model.model_validate({**data, "form_type": form_type})
    except ValidationError as exc:
        logger.warning("양식 검증 실패: form_type=%s errors=%s", form_type, exc.errors(include_url=False))
        raise ToolError("FORM_INVALID", f"{form_type} 초안이 양식 스키마에 맞지 않습니다.", tool="fill_form") from exc
    return {"form_type": form_type, "data": validated.model_dump()}
