"""작업 등록 API의 의존성 (FastAPI Depends).

라우터는 아래 `Annotated` 별칭을 파라미터 타입으로 쓰면 된다.

    def work_register(body: WorkRegisterRequest, agent: WorkRegisterAgentDep): ...

엔진·모델은 서버 설정(config.py의 SCENARIO_MODELS 표, 기동 때 만든 app.state.llm_provider)으로만 정한다.
요청자가 엔진을 고를 수 없다 (지침 4장 FastAPI 주의).
도구 레지스트리는 기동 후 한 번만 만든다 (인자 없는 lru_cache 함수).
테스트는 `app.dependency_overrides[get_work_register_agent]`로 바꾼다. override 키는 이 모듈의 함수 객체와 같아야 한다.
"""

from __future__ import annotations
from functools import lru_cache
import logging
from typing import Annotated
from fastapi import Depends, Request
from app.tools.tool_registry import ToolRegistry
from app.publicdata import register_publicdata_msds
from ..tools.tool_setup import build_tool_registry
from .work_register import WorkRegisterAgent
from app.core import Settings, get_settings
from app.llm.llm_provider import LLMClientProvider
from app.services.agent_work_service import WorkRegisterAgentFactory
from app.agent.record_query import RecordQueryAgent
from app.integrations import InternalApiClient
from app.tools.record_search import SEARCH_RECORDS_PARAMETERS, make_search_records_tool
from app.exceptions import ToolError

logger = logging.getLogger(__name__)

@lru_cache
def get_tool_registry() -> ToolRegistry:
    # A의 도구는 기동 시 한 번 만든다. E의 search_records는 요청별 Spring 토큰이
    # 필요하므로 get_record_query_agent()에서 별도로 등록한다.
    return build_tool_registry(
        rag_backend=get_settings().rag_backend,
        extra=register_publicdata_msds,
    )

ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]

def get_llm_provider(request: Request) -> LLMClientProvider:
    """기동 때 create_app()이 app.state.llm_provider에 넣어 둔 시나리오별 모델 제공자."""
    return request.app.state.llm_provider

LLMProviderDep = Annotated[LLMClientProvider, Depends(get_llm_provider)]

def get_work_register_agent(
    settings: Annotated[Settings, Depends(get_settings)],
    registry: ToolRegistryDep,
    provider: LLMProviderDep,
) -> WorkRegisterAgent:
    """개발 확인용 경로(/api/v1/test/work-register). 시나리오 A 모델을 쓴다."""
    # 엔진 문제(EngineBlockedError · LLMConfigError)는 여기서 나고 llm_error_handler가 응답을 만든다.
    return WorkRegisterAgent(llm=provider.get("A"), tools=registry, judge_max_tokens=settings.ai_max_output_tokens)

WorkRegisterAgentDep = Annotated[WorkRegisterAgent, Depends(get_work_register_agent)]

def get_record_query_agent(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: LLMProviderDep,
) -> RecordQueryAgent:
    # Spring이 AI 호출 때 보낸 토큰을 같은 요청의 search_records 호출에 전달한다.
    forwarded_token = request.headers.get("X-Internal-Token", "")
    if not forwarded_token:
        raise ToolError("TOOL_UNAVAILABLE", "Spring 내부 인증 헤더가 없습니다.", tool="search_records")
    logger.info("record-query 내부 인증 전달: scenario=E tokenProvided=%s", bool(forwarded_token))
    registry = build_tool_registry(
        rag_backend=settings.rag_backend,
        extra=register_publicdata_msds,
    )
    client = InternalApiClient(
        settings.internal_api_base_url,
        forwarded_token,
        connect_timeout=settings.internal_api_connect_timeout_seconds,
        read_timeout=settings.internal_api_read_timeout_seconds,
    )
    registry.register(
        "search_records",
        make_search_records_tool(client),
        owner="D",
        description="Spring 현장 기록 조회",
        parameters=SEARCH_RECORDS_PARAMETERS,
        replace=True,
    )
    binding = provider.binding_for("E")
    logger.info(
        "record-query 에이전트 선택: scenario=E engine=%s model=%s",
        binding.engine,
        binding.model_id,
    )
    return RecordQueryAgent(llm=provider.get("E"), tools=registry)

RecordQueryAgentDep = Annotated[RecordQueryAgent, Depends(get_record_query_agent)]

def get_work_register_agent_factory(
    settings: Annotated[Settings, Depends(get_settings)],
    registry: ToolRegistryDep,
    provider: LLMProviderDep,
) -> WorkRegisterAgentFactory:
    """Spring 실행 경로. 서비스가 요청의 mode로 시나리오(A·F)를 정한 뒤 에이전트를 만든다."""
    def make(scenario_key: str) -> WorkRegisterAgent:
        return WorkRegisterAgent(
            llm=provider.get(scenario_key), tools=registry, judge_max_tokens=settings.ai_max_output_tokens
        )
    return make

WorkRegisterAgentFactoryDep = Annotated[WorkRegisterAgentFactory, Depends(get_work_register_agent_factory)]
