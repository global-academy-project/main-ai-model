"""Spring이 부르는 에이전트 실행 경로 (라우터만).

    POST /api/v1/agent/work-register   {runId, siteId, permitId?, message, slots, mode}
        채팅 입구다. 시나리오 분류(router 모델) → 현장 모드 보정 → 처리기(A·F는 작업 등록 에이전트)
        성공 200  {success: true, data: WorkRegisterResult}   → Spring이 agent_runs.output_json에 저장
        실패      {success: false, error: {...}, steps: [...]} (agent_run_handlers)

/api/v1/test/* 는 개발 확인용으로 남기고 통합 때 막는다 (계획서 7-0).
"""

from fastapi import APIRouter

from app.agent.scenario.scenario_deps import ScenarioDispatcherDep
from app.schemas import DataResponse
from app.schemas.agent_work import AgentWorkRegisterRequest, WorkRegisterResult
from app.services import agent_work_service

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/work-register", response_model=DataResponse[WorkRegisterResult])
def work_register(
    body: AgentWorkRegisterRequest,
    dispatcher: ScenarioDispatcherDep,
) -> DataResponse[WorkRegisterResult]:
    # 동기 LLM 호출이 있으므로 def (스레드풀에서 실행)
    return agent_work_service.run_agent_turn(dispatcher, body)
