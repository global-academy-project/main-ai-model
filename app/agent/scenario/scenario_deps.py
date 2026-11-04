"""시나리오 분류 API의 의존성. 인자 없는 `Depends` 함수만 둔다 (요청자가 값을 고르지 못하게)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.agent_deps import ToolRegistryDep
from app.agent.scenario.scenario_classifier import ScenarioClassifier
from app.agent.scenario.scenario_dispatch import ScenarioDispatcher
from app.agent.scenario.scenario_handlers import load_chat_scenarios, register_work_register_handlers
from app.agent.scenario.scenario_retriever import ReferenceRetriever
from app.agent.scenario.scenario_router import Scenario
from app.core.config import get_settings
from app.llm.llm_provider import LLMClientProvider


def get_llm_provider(request: Request) -> LLMClientProvider:
    """기동 때 `create_app()`에서 `app.state.llm_provider`에 넣어 둔 제공자."""
    return request.app.state.llm_provider


LLMProviderDep = Annotated[LLMClientProvider, Depends(get_llm_provider)]


def get_chat_scenarios(request: Request) -> tuple[Scenario, ...] | None:
    """채팅 범위 (None이면 전부). 기동 때 `create_app()`이 검사해 `app.state.chat_scenarios`에 넣어 둔 값을 쓴다.

    `create_app()`을 거치지 않은 앱(테스트용)이면 같은 검사(`load_chat_scenarios`)로 설정에서 읽는다.
    """
    state = request.app.state
    if hasattr(state, "chat_scenarios"):
        return state.chat_scenarios
    return load_chat_scenarios(get_settings().chat_scenarios)


ChatScenariosDep = Annotated[tuple[Scenario, ...] | None, Depends(get_chat_scenarios)]


def get_scenario_classifier(
    provider: LLMProviderDep, registry: ToolRegistryDep, chat: ChatScenariosDep
) -> ScenarioClassifier:
    return ScenarioClassifier(provider, ReferenceRetriever(registry), chat_scenarios=chat)


ScenarioClassifierDep = Annotated[ScenarioClassifier, Depends(get_scenario_classifier)]


def get_scenario_dispatcher(
    provider: LLMProviderDep, registry: ToolRegistryDep, chat: ChatScenariosDep
) -> ScenarioDispatcher:
    """통합 채팅용. 분류(참고 자료 검색 포함)와 A·F 처리기를 묶어 요청마다 만든다 (모델은 provider가 공유)."""
    settings = get_settings()
    classifier = ScenarioClassifier(provider, ReferenceRetriever(registry), chat_scenarios=chat)
    dispatcher = ScenarioDispatcher(provider, chat_scenarios=chat, classifier=classifier)
    # /work-register(agent_deps.get_work_register_agent)와 같은 출력 토큰 한도를 쓴다
    register_work_register_handlers(dispatcher, registry, judge_max_tokens=settings.ai_max_output_tokens)
    return dispatcher


ScenarioDispatcherDep = Annotated[ScenarioDispatcher, Depends(get_scenario_dispatcher)]
