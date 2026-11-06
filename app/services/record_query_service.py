"""기록질문 서비스. FastAPI 세부 구현을 모른다."""

from app.agent.record_query import RecordQueryAgent
from app.schemas.base_schemas import DataResponse
from app.schemas.record_query import RecordQueryRequest, RecordQueryResult


def answer_record_query(
    agent: RecordQueryAgent, body: RecordQueryRequest
) -> DataResponse[RecordQueryResult]:
    return DataResponse[RecordQueryResult](data=agent.run(body))
