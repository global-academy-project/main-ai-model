"""Spring 이 부르는 문장 → 규칙 입력 LLM 추출 경로 (라우터만).

    POST /api/v1/extract/rule-input   {text, measurementItems, workTypes}
        성공 200  {success: true, data: {measurements, workTypes, watcherName, watcherEvidence}}
        실패      {success: false, error: {...}} (llm_error_handler · validation_error_handler)

Spring 은 실패하면 사전 추출만 쓴다 (APP_EXTRACT_LLM_CLIENT=fastapi 일 때만 부른다, 대기 15초).
등록: app/main.py 에 app.include_router(rule_input_extract_router) (B님).
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.agent.agent_deps import LLMProviderDep
from app.core import Settings, get_settings
from app.schemas import DataResponse
from app.schemas.rule_input_extract import RuleInputExtraction, RuleInputExtractRequest
from app.services.rule_input_extract_service import SCENARIO_KEY, run_rule_input_extract

router = APIRouter(prefix="/api/v1/extract", tags=["extract"])


@router.post("/rule-input", response_model=DataResponse[RuleInputExtraction])
def rule_input(
        body: RuleInputExtractRequest,
        provider: LLMProviderDep,
        settings: Annotated[Settings, Depends(get_settings)],
) -> DataResponse[RuleInputExtraction]:
    # 동기 LLM 호출이 있으므로 def (스레드풀에서 실행). 엔진 문제는 provider.get 에서 나고 llm_error_handler 가 받는다
    return run_rule_input_extract(provider.get(SCENARIO_KEY), body, max_tokens=settings.ai_max_output_tokens)
