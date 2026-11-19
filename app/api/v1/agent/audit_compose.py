"""감사 대응 보고서 구성기 전용 API (LLM 없음).

기존 `/api/v1/agent/audit-report`(D)와 나란히 두며, 서로 호출하지 않는다.
이 라우터는 아직 `main.py`에 연결하지 않았다(2단계, B 소유 공유 파일).
"""

from fastapi import APIRouter

from app.agent.audit_compose import compose_audit_report
from app.schemas.audit_compose import AuditComposeInput, AuditComposeResult
from app.schemas.base_schemas import DataResponse

router = APIRouter(prefix="/api/v1/agent", tags=["audit-compose"])


@router.post("/audit-report/compose", response_model=DataResponse[AuditComposeResult])
def audit_report_compose(body: AuditComposeInput) -> DataResponse[AuditComposeResult]:
    """서버가 집계한 항목을 묶음·상태값·서술문·종합 내역으로 구성한다. 조회·LLM 호출은 하지 않는다."""
    return DataResponse(data=compose_audit_report(body))
