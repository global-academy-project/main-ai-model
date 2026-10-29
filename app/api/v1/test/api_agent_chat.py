"""통합 채팅 라우터 (임시 prefix /api/v1/test). 경로만 둔다. 변환·분기는 서비스가 한다."""

from __future__ import annotations

from fastapi import APIRouter

from app.agent.scenario.scenario_deps import ScenarioDispatcherDep
from app.schemas.agent_chat import AgentChatRequest, AgentChatResponse
from app.schemas.base_schemas import DataResponse
from app.services.agent_chat_service import chat

router = APIRouter(prefix="/api/v1/test", tags=["agent-chat"])


@router.post("/agent-chat", response_model=DataResponse[AgentChatResponse])
def agent_chat(body: AgentChatRequest, dispatcher: ScenarioDispatcherDep) -> DataResponse[AgentChatResponse]:
    return chat(dispatcher, body)
