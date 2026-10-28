"""작업 등록 API의 의존성 (FastAPI Depends).

라우터는 아래 `Annotated` 별칭을 파라미터 타입으로 쓰면 된다.

    def work_register(body: WorkRegisterRequest, agent: WorkRegisterAgentDep): ...

엔진은 서버 설정(default_llm_engine)으로만 정한다. 요청자가 엔진을 고를 수 없다 (지침 4장 FastAPI 주의).
도구 레지스트리는 기동 후 한 번만 만든다 (인자 없는 lru_cache 함수).
테스트는 `app.dependency_overrides[get_work_register_agent]`로 바꾼다. override 키는 이 모듈의 함수 객체와 같아야 한다.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Annotated
from fastapi import Depends
from app.agent.tool.tool_registry import ToolRegistry
from .tool.tool_setup import build_tool_registry
from .work_register import WorkRegisterAgent
from app.core import Settings, get_settings
from app.llm.llm_factory import build_llm_client

@lru_cache
def get_tool_registry() -> ToolRegistry:
    return build_tool_registry()

ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]

def get_work_register_agent(
    settings: Annotated[Settings, Depends(get_settings)],
    registry: ToolRegistryDep,
) -> WorkRegisterAgent:
    # 엔진 문제(EngineBlockedError · LLMConfigError)는 여기서 나고 llm_error_handler가 응답을 만든다.
    llm = build_llm_client(settings=settings)
    return WorkRegisterAgent(llm=llm, tools=registry, judge_max_tokens=settings.ai_max_output_tokens)

WorkRegisterAgentDep = Annotated[WorkRegisterAgent, Depends(get_work_register_agent)]
