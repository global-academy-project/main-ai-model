"""작업 등록 에이전트 서비스: 에이전트 실행 결과를 응답 형식(DataResponse)으로 돌려준다.

FastAPI를 모른다 (Depends, Request 없음). 에이전트와 레지스트리는 인자로 받는다.
라우터는 이 함수의 반환값을 그대로 내보낸다.
"""

from app.agent.tool.tool_registry import ToolRegistry
from app.agent.work_register import WorkRegisterAgent
from app.schemas.agent_work import ToolInfo, WorkRegisterRequest, WorkRegisterResult
from app.schemas.base_schemas import DataResponse

def register_work(agent: WorkRegisterAgent, body: WorkRegisterRequest) -> DataResponse[WorkRegisterResult]:
    """발화 하나를 처리한다 (되묻기 또는 초안). LLMError·ToolError는 그대로 올려 핸들러가 응답으로 바꾼다."""
    return DataResponse[WorkRegisterResult](data=agent.run(body))

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
