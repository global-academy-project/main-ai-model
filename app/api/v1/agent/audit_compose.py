"""감사 대응 보고서 구성기 전용 API.

기존 `/api/v1/agent/audit-report`(D)와 나란히 두며, 서로 호출하지 않는다.
 Spring 백엔드 호출은 아직 연결하지 않았다.
"""

from fastapi import APIRouter, Request

from app.agent.audit_compose import compose_audit_report
from app.agent.audit_narrative import generate_sentence_candidates
from app.core import get_settings
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult
from app.schemas.base_schemas import DataResponse

router = APIRouter(prefix="/api/v1/agent", tags=["audit-compose"])


@router.post("/audit-report/compose", response_model=DataResponse[AuditComposeResult])
def audit_report_compose(
    body: AuditComposeInput,
    request: Request,
) -> DataResponse[AuditComposeResult]:
    """서버 집계 항목을 구성한다. P2 LLM 문장은 설정이 켜진 경우에만 사용한다."""
    settings = get_settings()
    generator = None
    if settings.audit_narrative_llm:
        # 시나리오 E(audit_query)의 D 소유 모델을 사용한다. 기존 summary 엔진의
        # 전용 메시지 후처리가 새 sentences JSON 형식에 끼어들지 않게 한다.
        llm = request.app.state.llm_provider.get("audit_query")
        generator = lambda item: generate_sentence_candidates(item, llm)
    return DataResponse(data=compose_audit_report(body, narrative_generator=generator))
