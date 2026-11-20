"""감사 대응 보고서 구성 서비스. FastAPI를 모른다 (Request, Depends 없음).

라우터(app/api/v1/agent/audit_compose.py)는 요청 본문·설정 값·모델을 얻는 방법만 넘기고,
이 서비스가 다음을 맡는다.
  - 항목 문장을 LLM으로 만들지 정한다 (AUDIT_NARRATIVE_LLM). 꺼져 있으면 모델을 얻으려 하지 않는다.
  - 모델을 못 얻으면(설정 오류 등) 5xx 대신 템플릿 문장으로 보고서를 만든다 (경고 로그 1줄, 원인 종류만).
  - 보고서 구성·숫자 대조는 app.agent.audit_compose가 한다 (그쪽은 LLM을 모른다).

모델은 값이 아니라 "얻는 함수"(get_llm)로 받는다. 문장 LLM이 꺼져 있으면 호출되지 않아서,
모델 제공자가 없는 앱·시험에서도 보고서가 만들어진다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.agent.audit_compose import compose_audit_report
from app.agent.audit_narrative import generate_sentence_candidates
from app.llm.llm_base import LLMClient
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult, ComposeItem
from app.schemas.base_schemas import DataResponse

__all__ = ["AUDIT_NARRATIVE_SCENARIO", "NarrativeGenerator", "build_narrative_generator", "compose_report"]

logger = logging.getLogger(__name__)

# 시나리오 E(audit_query)의 D 소유 모델을 사용한다. 기존 summary 엔진의
# 전용 메시지 후처리가 새 sentences JSON 형식에 끼어들지 않게 한다.
AUDIT_NARRATIVE_SCENARIO = "audit_query"

NarrativeGenerator = Callable[[ComposeItem], list[str]]


def build_narrative_generator(get_llm: Callable[[], LLMClient]) -> NarrativeGenerator | None:
    """문장 후보 생성기를 만든다. 모델을 못 얻으면 None (템플릿 문장만 쓴다)."""
    try:
        llm = get_llm()
    except Exception as exc:  # 모델 설정 오류가 보고서 전체를 5xx로 만들지 않게 한다
        logger.warning("감사 보고서 문장 모델을 못 얻어 템플릿 문장을 사용함 (%s)", type(exc).__name__)
        return None
    return lambda item: generate_sentence_candidates(item, llm)


def compose_report(
    body: AuditComposeInput,
    *,
    narrative_llm_enabled: bool,
    get_llm: Callable[[], LLMClient],
) -> DataResponse[AuditComposeResult]:
    """서버 집계 항목을 구성한다. 항목 문장 LLM은 narrative_llm_enabled가 켜진 경우에만 쓴다."""
    generator = build_narrative_generator(get_llm) if narrative_llm_enabled else None
    return DataResponse(data=compose_audit_report(body, narrative_generator=generator))
