"""통합 채팅 API 요청·응답 (POST /api/v1/test/agent-chat).

FastAPI는 대화 상태를 저장하지 않는다. 되묻기 응답이 "다음 요청에 무엇을 돌려보낼지"를 알려 준다.

    askKind = null        끝남 (result에 초안). 돌려보낼 것 없음
    askKind = "SCENARIO"  어떤 일인지 몰라 되물음. 다음 요청에 pendingText를 그대로 돌려보낸다
                          (서버가 이번 답과 이어 붙여 다시 분류한다)
    askKind = "SLOT"      작업 항목이 빠져 되물음. 다음 요청에 inProgress와 result.slots를 돌려보낸다
                          (분류 모델 없이 같은 처리기로 이어 간다)

전용 화면은 `scenario`를 직접 지정한다 (분류하지 않는다). 엔진·모델은 요청으로 정하지 않는다 (서버 설정 SCENARIO_ENGINES).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .agent_work import WorkRegisterResult, WorkSlots
from .base_schemas import CamelModel

ScenarioCode = Literal["A", "D", "E", "F"]
AskKind = Literal["SCENARIO", "SLOT"]
MAX_MESSAGE_CHARS = 2000


class AgentChatRequest(CamelModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, description="이번 발화")
    scenario: ScenarioCode | None = Field(default=None, description="전용 화면이 시나리오를 직접 지정. 있으면 분류하지 않는다")
    in_progress: ScenarioCode | None = Field(default=None, description="askKind=SLOT 응답에서 받은 값을 그대로")
    pending_text: str | None = Field(
        default=None, max_length=MAX_MESSAGE_CHARS, description="askKind=SCENARIO 응답에서 받은 값을 그대로"
    )
    slots: WorkSlots | None = Field(default=None, description="이전 응답 result.slots. 첫 요청이면 비운다")
    mode: Literal["VERIFY", "GUIDE"] = Field(default="VERIFY", description="sites.mode. F는 항상 GUIDE로 처리된다")

    @model_validator(mode="after")
    def _scenario_or_in_progress(self) -> AgentChatRequest:
        if self.scenario is not None and self.in_progress is not None:
            raise ValueError("scenario와 in_progress는 함께 보낼 수 없습니다.")
        return self


class ModelInfo(CamelModel):
    engine: str
    model_id: str | None = None


class AgentChatResponse(CamelModel):
    scenario: str = Field(description='"A"~"F" 또는 "UNKNOWN"')
    label: str | None = None
    source: Literal["llm", "fallback", "continuation", "explicit", "empty"]
    confidence: float
    owner: str
    handled: bool = Field(description="처리기가 돌았다. false면 아직 처리기가 없는 시나리오(owner가 담당)이거나 되묻는 중")
    ask_kind: AskKind | None = None
    ask_message: str | None = Field(default=None, description="화면에 보일 되묻는 말")
    pending_text: str | None = Field(default=None, description="askKind=SCENARIO일 때 다음 요청에 돌려보낼 값")
    in_progress: ScenarioCode | None = Field(default=None, description="askKind=SLOT일 때 다음 요청에 돌려보낼 값")
    redirect: str | None = Field(default=None, description="채팅 범위 밖이라 전용 화면으로 안내할 시나리오")
    model: ModelInfo | None = Field(default=None, description="처리에 쓴 엔진·모델 (진단용)")
    result: WorkRegisterResult | None = None
