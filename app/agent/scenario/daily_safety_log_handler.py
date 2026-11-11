from __future__ import annotations

from app.agent.daily_safety_log import DailySafetyLogAgent
from app.schemas.daily_safety_log import DailySafetyLogInput

from .scenario_dispatch import ScenarioHandler, ScenarioRequest


def make_daily_safety_log_handler() -> ScenarioHandler:
    def handler(request: ScenarioRequest, llm):
        raw = request.payload.get("daily_safety_log")
        if not isinstance(raw, dict):
            raise ValueError("시나리오 D에는 daily_safety_log 데이터가 필요합니다.")
        return DailySafetyLogAgent(llm=llm).run(DailySafetyLogInput.model_validate(raw))

    return handler
