"""Spring이 호출하는 기록질문 API 라우터."""

from fastapi import APIRouter

from app.api.v1.agent.record_query_deps import RecordQueryAgentDep
from app.schemas.base_schemas import DataResponse
from app.schemas.record_query import RecordQueryRequest, RecordQueryResult
from app.services.record_query_service import answer_record_query

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/record-query", response_model=DataResponse[RecordQueryResult])
def record_query(
    body: RecordQueryRequest, agent: RecordQueryAgentDep
) -> DataResponse[RecordQueryResult]:
    return answer_record_query(agent, body)
