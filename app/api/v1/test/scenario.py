"""시나리오 분류 API (라우터만).

    POST /api/v1/test/scenario-classify   말 → 시나리오 + 담당 모델 (+ 되묻는 문장)
    GET  /api/v1/test/scenario-engines    시나리오별로 쓰일 엔진·모델

분류기는 의존성(app/agent/scenario_deps.py), 응답은 서비스(app/services/scenario_service.py)가 만든다.
LLMError·ToolError는 전역 예외 핸들러가 응답으로 바꾼다.
"""

from fastapi import APIRouter

from app.agent.scenario.scenario_deps import LLMProviderDep, ScenarioClassifierDep
from app.schemas import DataResponse
from app.schemas.scenario import ModelInfo, ScenarioClassifyRequest, ScenarioClassifyResult
from app.services import scenario_service

router = APIRouter(prefix="/api/v1/test", tags=["test"])

@router.post("/scenario-classify", response_model=DataResponse[ScenarioClassifyResult])
def scenario_classify(
    body: ScenarioClassifyRequest, classifier: ScenarioClassifierDep
) -> DataResponse[ScenarioClassifyResult]:
    # 동기 모델 호출이 있으므로 def (스레드풀에서 실행)
    return scenario_service.classify_scenario(classifier, body)

@router.get("/scenario-engines", response_model=DataResponse[dict[str, ModelInfo]])
def scenario_engines(provider: LLMProviderDep) -> DataResponse[dict[str, ModelInfo]]:
    return scenario_service.list_scenario_engines(provider)
