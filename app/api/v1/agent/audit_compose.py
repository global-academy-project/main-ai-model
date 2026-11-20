"""감사 대응 보고서 구성기 API.

기존 `/api/v1/agent/audit-report` 주소를 유지하고, `/compose`는 호환용 별칭으로 제공한다.
집계는 호출자가 전달하며 이 API는 결과를 구성하고 검증한다.
"""

from fastapi import APIRouter, Request

from app.core import get_settings
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult
from app.schemas.base_schemas import DataResponse
from app.services.audit_compose_service import AUDIT_NARRATIVE_SCENARIO, compose_report

router = APIRouter(prefix="/api/v1/agent", tags=["audit-compose"])

@router.post("/audit-report", response_model=DataResponse[AuditComposeResult])
def audit_report_compose(
    body: AuditComposeInput,
    request: Request,
) -> DataResponse[AuditComposeResult]:
    """서버 집계 항목을 구성한다. P2 LLM 문장은 설정이 켜진 경우에만 사용한다 (선택·실패 처리는 서비스)."""
    return compose_report(
        body,
        narrative_llm_enabled=get_settings().audit_narrative_llm,
        # 꺼져 있으면 호출되지 않는다 (모델 제공자가 없는 앱에서도 보고서는 만들어진다).
        get_llm=lambda: request.app.state.llm_provider.get(AUDIT_NARRATIVE_SCENARIO),
    )
