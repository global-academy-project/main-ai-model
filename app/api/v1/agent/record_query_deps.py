"""기록질문 경로 전용 의존성. 공용 agent_deps와 레지스트리를 변경하지 않는다."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.agent.record_query import RecordQueryAgent
from app.api.llm_deps import scenario_llm
from app.core import Settings, get_settings
from app.llm.llm_base import LLMClient
from app.tools.record_search import make_search_records_tool
from app.integrations import InternalApiClient

RecordQueryLLM = Annotated[LLMClient, Depends(scenario_llm("audit_query"))]


def get_record_query_agent(
    llm: RecordQueryLLM,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RecordQueryAgent:
    if settings.internal_api_base_url is None or settings.internal_api_token is None:
        raise RuntimeError("기록질문에는 INTERNAL_API_BASE_URL과 INTERNAL_API_TOKEN이 필요합니다.")
    client = InternalApiClient(
        settings.internal_api_base_url,
        settings.internal_api_token.get_secret_value(),
        connect_timeout=2.0,
        read_timeout=settings.ai_timeout_seconds,
    )
    return RecordQueryAgent(llm=llm, search_records=make_search_records_tool(client))


RecordQueryAgentDep = Annotated[RecordQueryAgent, Depends(get_record_query_agent)]
