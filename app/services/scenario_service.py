"""시나리오 분류 API의 서비스: 분류기를 부르고 응답(DataResponse)을 만든다. FastAPI를 모른다."""

from __future__ import annotations

from app.agent.scenario.scenario_classifier import ClassificationResult, ScenarioClassifier
from app.agent.scenario.scenario_router import LABELS
from app.llm.llm_provider import EngineBinding, LLMClientProvider
from app.schemas import DataResponse
from app.schemas.scenario import (
    ModelInfo,
    ReferenceFailureInfo,
    ReferenceInfo,
    ScenarioClassifyRequest,
    ScenarioClassifyResult,
)

def _model(binding: EngineBinding | None) -> ModelInfo | None:
    return None if binding is None else ModelInfo(engine=binding.engine, model_id=binding.model_id)

def _to_result(result: ClassificationResult) -> ScenarioClassifyResult:
    d = result.decision
    return ScenarioClassifyResult(
        scenario=d.scenario.value,
        label=LABELS.get(d.scenario),
        confidence=d.confidence,
        reason=d.reason,
        source=d.source,
        candidate=d.candidate.value if d.candidate else None,
        redirect=d.redirect.value if d.redirect else None,
        ask=d.ask,
        owner=d.owner,
        handled_by_b=d.handled_by_b,
        classifier_model=_model(result.classifier_model),
        scenario_model=_model(result.scenario_model),
        rag_used=result.rag_used,
        references=[
            ReferenceInfo(id=r.id, corpus=r.corpus, source=r.source, score=r.score, text=r.text)
            for r in result.references
        ],
        reference_failures=[
            ReferenceFailureInfo(corpus=f.corpus, code=f.code, reason=f.reason) for f in result.reference_failures
        ],
    )


def classify_scenario(
    classifier: ScenarioClassifier, body: ScenarioClassifyRequest
) -> DataResponse[ScenarioClassifyResult]:
    result = classifier.classify(body.message, use_rag=body.use_rag)
    return DataResponse(success=True, data=_to_result(result))


def list_scenario_engines(provider: LLMClientProvider) -> DataResponse[dict[str, ModelInfo]]:
    """키(router, A~F)별로 실제 쓰이는 엔진·모델 (진단용. 모델을 읽지는 않는다)."""
    described = {
        key: ModelInfo(engine=value["engine"] or "", model_id=value["model_id"])
        for key, value in provider.describe().items()
    }
    return DataResponse(success=True, data=described)
