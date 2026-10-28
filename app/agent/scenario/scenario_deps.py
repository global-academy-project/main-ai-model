"""시나리오 분류 API의 의존성. 인자 없는 `Depends` 함수만 둔다 (요청자가 값을 고르지 못하게)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.agent_deps import ToolRegistryDep
from app.agent.scenario.scenario_classifier import ScenarioClassifier
from app.agent.scenario.scenario_dispatch import parse_chat_scenarios
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
