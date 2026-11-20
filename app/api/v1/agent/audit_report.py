"""서버 집계값으로 감사보고서 초안을 만드는 전용 API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.audit_report import AuditReportAgent
from app.api.llm_deps import audit_report_llm
from app.llm.llm_base import LLMClient
from app.schemas.audit_report import AuditReportInput, AuditReportResult
from app.schemas.base_schemas import DataResponse

# router = APIRouter(prefix="/api/v1/agent", tags=["audit-report"])
# AuditReportLLM = Annotated[LLMClient, Depends(audit_report_llm)]


# @router.post("/audit-report", response_model=DataResponse[AuditReportResult])
# def audit_report(
#     body: AuditReportInput,
#     llm: AuditReportLLM,
# ) -> DataResponse[AuditReportResult]:
#     """SQL 집계는 수행하지 않고, 서버가 전달한 stats를 요약·검증한다."""
#     result = AuditReportAgent(llm=llm).run(body)
#     return DataResponse(data=result)
