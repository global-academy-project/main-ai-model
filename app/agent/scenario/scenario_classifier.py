"""시나리오 분류 한 번: (선택) 참고 자료 검색 → 라우터 분류 → 어느 모델로 갈지 붙이기.

    메시지 ─ 검색 도구(mock 포함)로 참고 자료 ─→ ScenarioRouter(router 모델) ─→ 시나리오 + 담당 모델

- 엔진을 쓰는 쪽이다. 엔진은 `LLMClientProvider`에서 받고, 어떤 엔진인지 모른다.
- 분류만 한다. 시나리오 처리기는 부르지 않는다 (처리까지 하려면 `ScenarioDispatcher`).
- 시나리오 모델은 **만들지 않는다**. 어떤 엔진·모델로 갈지(`binding_for`)만 알려 준다 (모델 로딩 없음).
- `LLMError`는 삼키지 않고 그대로 올린다 (HTTP 핸들러가 처리).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.llm.llm_provider import ROUTER_KEY, EngineBinding, LLMClientProvider

from ..evidence_collector.models import SearchFailure
from .scenario_retriever import Reference, ReferenceRetriever
from .scenario_router import MAX_REFERENCES_IN_PROMPT, Scenario, ScenarioDecision, ScenarioRouter


@dataclass(frozen=True)
class ClassificationResult:
    decision: ScenarioDecision
    classifier_model: EngineBinding  # 분류에 쓴 엔진·모델
    scenario_model: EngineBinding | None  # 이 시나리오를 처리할 엔진·모델. UNKNOWN이면 None
    rag_used: bool  # 검색을 시도했나
    references: tuple[Reference, ...] = ()  # 분류 프롬프트에 실제로 들어간 후보 (앞 3개까지)
    reference_failures: tuple[SearchFailure, ...] = ()


class ScenarioClassifier:
    def __init__(
        self,
        provider: LLMClientProvider,
        retriever: ReferenceRetriever | None = None,
        *,
        chat_scenarios: Iterable[Scenario] | None = None,
    ) -> None:
        """chat_scenarios: 통합 채팅창에서 분류해 줄 시나리오 (기본 전부). 잘못된 값은 ValueError."""
        self._provider = provider
        self._retriever = retriever
        self._chat_scenarios = None if chat_scenarios is None else tuple(chat_scenarios)

    def classify(self, message: str, *, use_rag: bool = True) -> ClassificationResult:
        """`use_rag=False`면 검색하지 않고 분류한다 (참고 자료 효과를 비교할 때)."""
        text = (message or "").strip()
        rag = use_rag and self._retriever is not None and bool(text)
        found = self._retriever.retrieve(text) if rag else None
        references = found.references if found else ()

        # 모델은 검색이 끝난 뒤에 만든다 (설정 문제·모델 로딩 실패는 여기서 LLMError로 나온다)
        llm = self._provider.get(ROUTER_KEY)
        router = ScenarioRouter(llm, allowed=self._chat_scenarios)
        decision = router.route(text, references=references)

        scenario_model = (
            None if decision.scenario is Scenario.UNKNOWN else self._provider.binding_for(decision.scenario.value)
        )
        return ClassificationResult(
            decision=decision,
            classifier_model=self._provider.binding_for(ROUTER_KEY),
            scenario_model=scenario_model,
            rag_used=rag,
            references=tuple(references[:MAX_REFERENCES_IN_PROMPT]),
            reference_failures=found.failures if found else (),
        )
