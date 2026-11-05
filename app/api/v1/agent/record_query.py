"""Spring이 호출하는 기록질문 API."""

import logging

from fastapi import APIRouter

from app.agent.agent_deps import RecordQueryAgentDep
from app.schemas import DataResponse
from app.schemas.record_query import RecordQueryRequest, RecordQueryResult

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])
logger = logging.getLogger(__name__)


@router.post("/record-query", response_model=DataResponse[RecordQueryResult])
def record_query(body: RecordQueryRequest, agent: RecordQueryAgentDep) -> DataResponse[RecordQueryResult]:
    logger.info(
        "record-query 시작: scenario=E runId=%s siteId=%s mode=%s",
        body.run_id,
        body.site_id,
        body.mode,
    )
    result = agent.run(body)
    logger.info(
        "record-query 완료: scenario=E runId=%s runStatus=%s queryCount=%s evidenceCount=%s",
        body.run_id,
        result.run_status,
        len(result.queries),
        len(result.evidence),
    )
    return DataResponse(data=result)
