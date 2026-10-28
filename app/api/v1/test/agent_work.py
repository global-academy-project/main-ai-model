"""작업 등록 에이전트 API (라우터만).

    POST /api/v1/test/work-register   발화 → 되묻기(WAITING_INPUT) 또는 초안(DONE)
    GET  /api/v1/test/tools           등록된 도구와 mock 여부

의존성은 app/api/agent_deps.py, 응답을 만드는 일은 app/services/agent_work_service.py에 있다.
이 파일은 경로와 ToolError 핸들러만 가진다.

ToolError → HTTP 매핑 (지침 6장 4번의 임시안. 코드 어휘는 3일차 규격에서 확정):
  EVIDENCE_SEARCH_FAILED, TOOL_UNAVAILABLE, 그 외   502
  TOOL_NOT_FOUND, FORM_INVALID, FORM_TYPE_UNKNOWN   500 (서버 쪽 설정·코드 문제)
  메시지는 ToolError.message(도구가 직접 쓴 고정 문구)를 그대로 쓴다.
"""

from fastapi import APIRouter
from app.schemas import DataResponse
from app.schemas.agent_work import ToolInfo, WorkRegisterRequest, WorkRegisterResult
from app.services import agent_work_service
from app.agent.agent_deps import ToolRegistryDep, WorkRegisterAgentDep

router = APIRouter(prefix="/api/v1/test", tags=["test"])

@router.post("/work-register", response_model=DataResponse[WorkRegisterResult])
def work_register(
    body: WorkRegisterRequest, 
    agent: WorkRegisterAgentDep
) -> DataResponse[WorkRegisterResult]:
    # 동기 LLM 호출이 있으므로 def (스레드풀에서 실행)
    return agent_work_service.register_work(agent, body)

@router.get("/tools", response_model=DataResponse[list[ToolInfo]])
def list_tools(registry: ToolRegistryDep) -> DataResponse[list[ToolInfo]]:
    return agent_work_service.list_registered_tools(registry)

