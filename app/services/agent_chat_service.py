"""통합 채팅 서비스: 디스패처 결과를 응답 DTO로 바꾼다. FastAPI를 모른다 (`Depends`, `Request` 없음).

되묻기 두 종류를 여기서 구분한다 (DTO 설명은 schemas/agent_chat.py).
  - 분류 단계의 되묻기(`decision.ask`)   → askKind="SCENARIO", pendingText에 다음 요청에 돌려보낼 말
  - 처리기가 WAITING_INPUT               → askKind="SLOT", inProgress에 다음 요청에 돌려보낼 시나리오
"""

from __future__ import annotations

from app.agent.scenario.scenario_dispatch import (
    PENDING_TEXT_KEY,
    DispatchResult,
    ScenarioDispatcher,
    ScenarioRequest,
    join_pending_text,
)
from app.agent.scenario.scenario_router import LABELS, Scenario
from app.schemas.agent_chat import AgentChatRequest, AgentChatResponse, ModelInfo
from app.schemas.agent_work import WorkRegisterResult
from app.schemas.base_schemas import DataResponse


def chat(dispatcher: ScenarioDispatcher, body: AgentChatRequest) -> DataResponse[AgentChatResponse]:
    request = ScenarioRequest(
        message=body.message,
        scenario=Scenario(body.scenario) if body.scenario else None,
        in_progress=Scenario(body.in_progress) if body.in_progress else None,
        payload={"slots": body.slots, "mode": body.mode, PENDING_TEXT_KEY: body.pending_text},
    )
    result = dispatcher.dispatch(request)
    return DataResponse(data=_to_response(result, body))


def _to_response(result: DispatchResult, body: AgentChatRequest) -> AgentChatResponse:
    decision = result.decision
    common = {
        "scenario": decision.scenario.value,
        "label": LABELS.get(decision.scenario),
        "source": decision.source,
        "confidence": decision.confidence,
        "owner": decision.owner,
        "handled": result.handled,
        "redirect": decision.redirect.value if decision.redirect else None,
    }

    # ① 분류 단계의 되묻기: 어떤 일인지 모른다. 이번 말까지 이어 붙여 돌려보내게 한다
    if result.ask is not None:
        return AgentChatResponse(
            **common,
            ask_kind="SCENARIO",
            ask_message=result.ask,
            pending_text=join_pending_text(body.pending_text, body.message),
        )

    if not result.handled:  # 아직 처리기가 없는 시나리오 (owner가 담당)
        return AgentChatResponse(**common)

    output = result.output
    if not isinstance(output, WorkRegisterResult):
        # B가 맡은 A·F만 연결돼 있다. 다른 시나리오 처리기를 등록하면 그 결과 DTO를 여기에 추가한다.
        raise TypeError(f"시나리오 {decision.scenario.value} 처리기의 반환 타입을 응답으로 바꿀 수 없습니다: {type(output).__name__}")

    model = ModelInfo(**result.model) if result.model else None
    # ② 처리기의 항목 되묻기: 같은 시나리오로 이어 가게 한다
    if output.run_status == "WAITING_INPUT":
        return AgentChatResponse(
            **common,
            ask_kind="SLOT",
            ask_message=output.ask.message if output.ask else None,
            in_progress=decision.scenario.value,
            model=model,
            result=output,
        )
    return AgentChatResponse(**common, model=model, result=output)
