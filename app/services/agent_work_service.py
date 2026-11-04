"""작업 등록 에이전트 서비스: 에이전트 실행 결과를 응답 형식(DataResponse)으로 돌려준다.

FastAPI를 모른다 (Depends, Request 없음). 에이전트와 레지스트리는 인자로 받는다.
라우터는 이 함수의 반환값을 그대로 내보낸다.

Spring 실행 경로(/api/v1/agent/work-register)는 채팅 입구다 (`run_agent_turn`):
    Spring 요청 → ScenarioRequest → 디스패처(이어 가기 확인 → router 분류 → 현장 모드 보정 → 처리기)
    → 항상 WorkRegisterResult 모양 (Spring이 data를 그대로 저장한다)

    slots 상태               이번 요청
    없음                     새 대화 → 분류
    pendingText만 있음        시나리오 되묻기의 답 → 앞선 말과 이어 붙여 분류
    utterances·askedKeys 있음 작업 등록을 이어 감 (분류 없이. A·F는 mode가 정한다)

    처리기가 돌지 않은 경우(분류 안 됨, 채팅 범위 밖, 처리기가 없는 시나리오)는
    runStatus=WAITING_INPUT + ask(되묻는 말·안내) + 빈 classification으로 돌려준다.

시나리오 모델: 사업장 모드(mode)로 고른다. VERIFY(일반) → 시나리오 A, GUIDE(안내) → 시나리오 F.
표는 app/agent/site_mode.py 한 곳에 있다 (채팅 경로도 같은 표를 쓴다).
어떤 엔진·모델인지는 app/core/config.py의 SCENARIO_MODELS 표가 정한다.
"""

import logging
from collections.abc import Callable, Sequence

from app.tools.tool_registry import ToolRegistry
from app.agent.scenario.scenario_dispatch import (
    PENDING_TEXT_KEY,
    SLOTS_KEY,
    DispatchResult,
    ScenarioDispatcher,
    ScenarioRequest,
    join_pending_text,
)
from app.agent.scenario.scenario_handlers import MODE_KEY, RUN_VALUE_KEYS, has_work_register_state
from app.agent.scenario.scenario_router import Scenario, build_ask
from app.agent.site_mode import SCENARIO_BY_MODE
from app.agent.work_register import WorkRegisterAgent
from app.schemas.agent_work import (
    AgentWorkRegisterRequest,
    AskBack,
    Classification,
    ToolInfo,
    WorkRegisterRequest,
    WorkRegisterResult,
    WorkSlots,
)
from app.schemas.base_schemas import DataResponse

__all__ = [
    "SCENARIO_BY_MODE",
    "WorkRegisterAgentFactory",
    "list_registered_tools",
    "register_work",
    "run_agent_turn",
    "run_work_register",
    "to_scenario_request",
    "to_work_register_result",
]

logger = logging.getLogger(__name__)

# 처리기가 돌지 않은 응답의 classification. 작업유형을 아직 모른다 (high_risk=None은 "모름").
_NOT_CLASSIFIED = Classification(work_types=[], high_risk=None, required_permits=[], can_start_before_approval=False)

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

def run_agent_turn(dispatcher: ScenarioDispatcher, body: AgentWorkRegisterRequest) -> DataResponse[WorkRegisterResult]:
    """Spring 실행 요청(채팅 한 번). 분류 → 처리기 → WorkRegisterResult.

    LLMError(분류·모델 로딩)는 그대로, 에이전트 실행 중 실패는 AgentRunError(steps 포함)로 올라간다.
    """
    request = to_scenario_request(body)
    result = dispatcher.dispatch(request)
    decision = result.decision
    # 사용자 말은 남기지 않는다. 분류가 모델로 됐는지(source=llm)·대체 규칙인지(fallback)를 여기서 본다.
    logger.info(
        "에이전트 실행 run=%s scenario=%s source=%s confidence=%.2f resolved_from=%s handled=%s model=%s rejected=%s",
        body.run_id,
        decision.scenario.value,
        decision.source,
        decision.confidence,
        result.resolved_from.value if result.resolved_from else None,
        result.handled,
        result.model,
        result.continuation_rejected,
    )
    return DataResponse[WorkRegisterResult](
        data=to_work_register_result(result, request, dispatcher.available_scenarios)
    )

def to_scenario_request(body: WorkRegisterRequest) -> ScenarioRequest:
    """Spring 요청 → 디스패처 요청. 대화 상태는 Spring이 돌려준 slots만 보고 정한다 (요청에 다른 상태 필드가 없다)."""
    payload: dict[str, object] = {MODE_KEY: body.mode, **{key: getattr(body, key) for key in RUN_VALUE_KEYS}}
    slots = body.slots
    if slots is not None and slots.pending_text:
        # 시나리오 되묻기의 답. 앞선 말과 이어 붙여 다시 분류한다 (작업 등록 상태는 아직 없다)
        return ScenarioRequest(message=body.message, payload={**payload, PENDING_TEXT_KEY: slots.pending_text})
    if has_work_register_state(slots):
        # 작업 등록을 이어 감. A·F 중 어느 쪽인지는 디스패처의 보정이 mode로 정한다
        return ScenarioRequest(
            message=body.message, in_progress=Scenario.WORK_REGISTER, payload={**payload, SLOTS_KEY: slots}
        )
    return ScenarioRequest(message=body.message, payload=payload)

def to_work_register_result(
    result: DispatchResult, request: ScenarioRequest, available: Sequence[Scenario]
) -> WorkRegisterResult:
    """디스패처 결과 → WorkRegisterResult. 처리기가 돌았으면 그 결과 그대로, 아니면 되묻기·안내."""
    if result.handled:
        if not isinstance(result.output, WorkRegisterResult):
            raise TypeError(
                f"시나리오 {result.decision.scenario.value} 처리기가 WorkRegisterResult가 아닌 값을 돌려줬습니다."
            )
        return result.output
    if result.ask is not None:
        if result.decision.redirect is not None:
            pending = None  # 다른 화면의 일이다. 다음 말은 새로 분류한다
        else:
            pending = join_pending_text(request.payload.get(PENDING_TEXT_KEY), request.message)
        return _ask_result(result.ask, pending)
    # 분류는 됐지만 처리기가 없는 시나리오 (예: D·E, 담당자가 아직 연결 전) → 전용 화면 안내
    return _ask_result(build_ask(redirect=result.decision.scenario, allowed=available), None)

def _ask_result(message: str, pending_text: str | None) -> WorkRegisterResult:
    return WorkRegisterResult(
        run_status="WAITING_INPUT",
        slots=WorkSlots(pending_text=pending_text),
        ask=AskBack(message=message, missing_fields=[]),
        classification=_NOT_CLASSIFIED,
    )

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
