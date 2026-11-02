"""작업 등록 에이전트 서비스: 에이전트 실행 결과를 응답 형식(DataResponse)으로 돌려준다.

FastAPI를 모른다 (Depends, Request 없음). 에이전트와 레지스트리는 인자로 받는다.
라우터는 이 함수의 반환값을 그대로 내보낸다.

시나리오 모델: 사업장 모드(mode)로 고른다. VERIFY(일반) → 시나리오 A, GUIDE(안내) → 시나리오 F.
표는 app/agent/site_mode.py 한 곳에 있다 (채팅 경로도 같은 표를 쓴다).
어떤 엔진·모델인지는 app/core/config.py의 SCENARIO_MODELS 표가 정한다.
"""

from collections.abc import Callable

from app.tools.tool_registry import ToolRegistry
from app.agent.site_mode import SCENARIO_BY_MODE
from app.agent.work_register import WorkRegisterAgent
from app.schemas.agent_work import AgentWorkRegisterRequest, ToolInfo, WorkRegisterRequest, WorkRegisterResult
from app.schemas.base_schemas import DataResponse

__all__ = ["SCENARIO_BY_MODE", "WorkRegisterAgentFactory", "list_registered_tools", "register_work", "run_work_register"]

# 시나리오 키("A"·"F")를 받아 그 시나리오 모델로 에이전트를 만든다 (agent_deps가 만든다)
WorkRegisterAgentFactory = Callable[[str], WorkRegisterAgent]

def register_work(agent: WorkRegisterAgent, body: WorkRegisterRequest) -> DataResponse[WorkRegisterResult]:
    """발화 하나를 처리한다 (되묻기 또는 초안). LLMError·ToolError는 그대로 올려 핸들러가 응답으로 바꾼다."""
    return DataResponse[WorkRegisterResult](data=agent.run(body))

def run_work_register(
    make_agent: WorkRegisterAgentFactory, body: AgentWorkRegisterRequest
) -> DataResponse[WorkRegisterResult]:
    """Spring 실행 요청. 모드에 맞는 시나리오 모델로 에이전트를 만들어 실행한다.

    실행 중 LLMError·ToolError는 AgentRunError(원인, steps)로 올라온다. 모델을 만들다 난 LLMError
    (설정·차단)는 실행 전이라 steps 없이 그대로 올라간다.
    """
    return register_work(make_agent(SCENARIO_BY_MODE[body.mode]), body)

def list_registered_tools(registry: ToolRegistry) -> DataResponse[list[ToolInfo]]:
    return DataResponse[list[ToolInfo]](
        data=[ToolInfo(
            name=t.name, 
            owner=t.owner, 
            mock=t.mock, 
            description=t.description
            ) for t in registry.tools()
        ]
    )
