"""문장 → 규칙 입력 LLM 추출 서비스 (C 담당, 계획서 7-4장 2단계). FastAPI 를 모른다.

LLM 호출 → JSON 꺼내기 → 검사(app/agent/rule_input_parse.py) → 응답(DataResponse).
- 모델: 시나리오 confined_space_gate (글자 코드 B, 엔진 app/llm/engines/llm_confined_space_gate.py, 주인 C).
- json_schema: 엔진의 chat() 이 이 인자를 받으면 넘긴다. 받지 않는 엔진(가짜·transformers)이면 빼고 부른다.
- 모델 출력이 JSON 이 아니면 API 는 실패로 돌려준다 (Spring 은 사전 추출 결과 + 안내를 보여 준다). 평가 스크립트는 빈 결과로 센다.
- LLMError 는 잡지 않는다. 기존 llm_error_handler 가 실패 응답(success: false)으로 바꾼다.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from app.agent.evidence_collector.textutil import extract_json_object
from app.agent.rule_input_parse import ExtractedRuleInput, parse_rule_input
from app.agent.rule_input_prompt import build_system_prompt, build_user_message, output_schema
from app.exceptions.llm_errors import LLMError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import Message
from app.schemas.base_schemas import DataResponse
from app.schemas.rule_input_extract import (
    ExtractedMeasurement,
    ExtractedWorkType,
    RuleInputExtraction,
    RuleInputExtractRequest,
)

__all__ = ["SCENARIO_KEY", "extract_rule_input", "run_rule_input_extract"]

log = logging.getLogger(__name__)

SCENARIO_KEY = "confined_space_gate"


def run_rule_input_extract(llm: LLMClient, body: RuleInputExtractRequest, *,
                           max_tokens: int | None = None) -> DataResponse[RuleInputExtraction]:
    out = extract_rule_input(llm, body.text, list(dict.fromkeys(body.measurement_items)),
                             list(dict.fromkeys(body.work_types)), max_tokens=max_tokens)
    if not out.parsed:
        raise LLMError("규칙 입력 추출: 모델 출력이 JSON 이 아님")   # Spring 은 사전 추출 결과 + 안내로 보여 준다
    data = RuleInputExtraction(
        measurements=[ExtractedMeasurement(item=m.item, value=float(m.value), unit=m.unit, evidence=m.evidence)
                      for m in out.measurements],
        work_types=[ExtractedWorkType(type=w.type, evidence=w.evidence) for w in out.work_types],
        watcher_name=out.watcher_name,
        watcher_evidence=out.watcher_evidence,
    )
    return DataResponse[RuleInputExtraction](data=data)


def extract_rule_input(llm: LLMClient, text: str, items: list[str], work_types: list[str], *,
                       max_tokens: int | None = None) -> ExtractedRuleInput:
    kwargs: dict[str, Any] = {"system": build_system_prompt(items, work_types), "max_tokens": max_tokens}
    if _accepts_json_schema(llm):
        kwargs["json_schema"] = output_schema(items, work_types)
    response = llm.chat([Message(role="user", content=build_user_message(text))], **kwargs)

    raw = extract_json_object(response.text or "")
    if raw is None:
        log.warning("규칙 입력 추출: 모델 출력이 JSON 이 아님 (stop_reason=%s)", response.stop_reason)
        return ExtractedRuleInput(parsed=False)
    out = parse_rule_input(raw, text, items, work_types)
    if out.dropped:
        log.info("규칙 입력 추출: 버린 값 %d개 %s", len(out.dropped), out.dropped)
    return out


def _accepts_json_schema(llm: LLMClient) -> bool:
    try:
        params = inspect.signature(llm.chat).parameters
    except (TypeError, ValueError):
        return False
    return "json_schema" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
