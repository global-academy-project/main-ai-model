from __future__ import annotations

import json
from typing import Any

from app.agent.evidence_collector import extract_json_object
from app.exceptions import AgentRunError
from app.llm.engines.llm_daily_safety_log import SYSTEM_PROMPT
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.daily_safety_log import DailySafetyLogInput, DailySafetyLogResult


class DailySafetyLogAgent:
    """Spring 없이 전달받은 공식 집계 데이터를 요약하는 D Agent."""

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, data: DailySafetyLogInput) -> DailySafetyLogResult:
        prompt = json.dumps(data.model_dump(by_alias=True), ensure_ascii=False)
        response = self._llm.chat([Message(role="user", content=prompt)], system=SYSTEM_PROMPT)
        try:
            raw = extract_json_object(response.text)
            if not isinstance(raw, dict):
                raise ValueError("JSON 객체가 아닙니다")
            return DailySafetyLogResult(
                site_id=data.site_id,
                date=data.date,
                summary=str(raw.get("summary", "확인되지 않음")),
                tbm=str(raw.get("tbm", "확인되지 않음")),
                deadline=str(raw.get("deadline", "확인되지 않음")),
                daily_safety_log=str(raw.get("dailySafetyLog", "확인되지 않음")),
                warnings=[str(v) for v in raw.get("warnings", []) if v is not None],
                evidence=[v for v in raw.get("evidence", []) if isinstance(v, dict)],
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentRunError("시나리오 D 모델 응답을 해석할 수 없습니다.") from exc
