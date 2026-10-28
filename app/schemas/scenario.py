"""시나리오 분류 API의 요청·응답 (HTTP 경계를 넘는 타입).

엔진 이름·모델은 **응답에만** 있다. 요청자가 엔진을 고를 수 없다 (서버 설정 `SCENARIO_ENGINES`로만).
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.base_schemas import CamelModel


class ScenarioClassifyRequest(CamelModel):
    message: str = Field(min_length=1, max_length=2000, description="사용자가 채팅창에 쓴 말")
    use_rag: bool = Field(default=True, description="False면 참고 자료 검색 없이 분류 (효과 비교용)")


class ModelInfo(CamelModel):
    engine: str
    model_id: str | None = None


class ReferenceInfo(CamelModel):
    id: str
    corpus: str
    source: str
    score: float
    text: str


class ReferenceFailureInfo(CamelModel):
    corpus: str
    code: str
    reason: str


class ScenarioClassifyResult(CamelModel):
    scenario: str  # A~F 또는 UNKNOWN
    label: str | None = None  # 화면에 보여 줄 이름 (UNKNOWN이면 없음)
    confidence: float
    reason: str
    source: str  # llm | fallback | empty  (fallback이면 모델이 아니라 키워드 규칙으로 정했다)
    candidate: str | None = None  # UNKNOWN이지만 가장 가까운 시나리오
    redirect: str | None = None  # 채팅 범위 밖이라 전용 화면으로 안내할 시나리오
    ask: str | None = None  # 분류가 안 됐을 때 되물을 문장
    owner: str  # 담당자 (B/C/D/A, 모르면 "-")
    handled_by_b: bool
    classifier_model: ModelInfo
    scenario_model: ModelInfo | None = None  # 처리 시 쓸 엔진·모델 (UNKNOWN이면 없음)
    rag_used: bool
    references: list[ReferenceInfo] = Field(default_factory=list)
    reference_failures: list[ReferenceFailureInfo] = Field(default_factory=list)
