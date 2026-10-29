"""시나리오 분류 API의 의존성. 인자 없는 `Depends` 함수만 둔다 (요청자가 값을 고르지 못하게)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.agent_deps import ToolRegistryDep
from app.agent.scenario.scenario_classifier import ScenarioClassifier
from app.agent.scenario.scenario_dispatch import ScenarioDispatcher, parse_chat_scenarios
from app.agent.scenario.scenario_handlers import register_work_register_handlers
from app.agent.scenario.scenario_retriever import ReferenceRetriever
from app.core.config import get_settings
from app.llm.llm_provider import LLMClientProvider


def get_llm_provider(request: Request) -> LLMClientProvider:
    """기동 때 `create_app()`에서 `app.state.llm_provider`에 넣어 둔 제공자."""
    return request.app.state.llm_provider


LLMProviderDep = Annotated[LLMClientProvider, Depends(get_llm_provider)]


def get_scenario_classifier(provider: LLMProviderDep, registry: ToolRegistryDep) -> ScenarioClassifier:
    chat = parse_chat_scenarios(getattr(get_settings(), "chat_scenarios", None))
    return ScenarioClassifier(provider, ReferenceRetriever(registry), chat_scenarios=chat)


ScenarioClassifierDep = Annotated[ScenarioClassifier, Depends(get_scenario_classifier)]


def get_scenario_dispatcher(provider: LLMProviderDep, registry: ToolRegistryDep) -> ScenarioDispatcher:
    """통합 채팅용. 분류(참고 자료 검색 포함)와 A·F 처리기를 묶어 요청마다 만든다 (모델은 provider가 공유)."""
    settings = get_settings()
    chat = parse_chat_scenarios(getattr(settings, "chat_scenarios", None))
    classifier = ScenarioClassifier(provider, ReferenceRetriever(registry), chat_scenarios=chat)
    dispatcher = ScenarioDispatcher(provider, chat_scenarios=chat, classifier=classifier)
    # /work-register(agent_deps.get_work_register_agent)와 같은 출력 토큰 한도를 쓴다
    register_work_register_handlers(dispatcher, registry, judge_max_tokens=settings.ai_max_output_tokens)
    return dispatcher


ScenarioDispatcherDep = Annotated[ScenarioDispatcher, Depends(get_scenario_dispatcher)]
