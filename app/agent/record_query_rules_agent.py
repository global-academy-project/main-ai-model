"""기록 질문 에이전트 (규칙 우선 계획). D의 `RecordQueryAgent`를 상속해 `_plan`만 앞단에서 가로챈다 (B).

D의 파일은 고치지 않는다. 규칙(`record_plan_rules`)이 정한 경우만 그 계획을 쓰고, 정하지 못하면 D의 계획(LLM)으로
그대로 넘긴다. 범위 밖 질문 판정(`_is_obviously_out_of_scope`)과 계획 검증(`_valid_params`)도 D의 것을 쓴다.
`app/main.py`가 RECORD_PLAN_RULES가 켜졌을 때만 이 클래스로 의존성을 바꿔 끼운다.
"""

from __future__ import annotations

import logging

from app.agent.record_plan_rules import rule_plan
from app.agent.record_query import PlanOutcome, RecordQueryAgent
from app.schemas.record_query import RecordQueryRequest

logger = logging.getLogger(__name__)


class RuleFirstRecordQueryAgent(RecordQueryAgent):
    def _plan(self, request: RecordQueryRequest) -> PlanOutcome:
        if self._is_obviously_out_of_scope(request.message):
            return super()._plan(request)  # D가 이미 모델 없이 처리한다
        plan = rule_plan(request.message, request.context)
        if plan is not None and self._valid_params(plan.query_type, plan.params):
            logger.info("기록 질문 계획: 규칙 run=%s type=%s", request.run_id, plan.query_type)
            return PlanOutcome(decision="QUERY", query_type=plan.query_type, params=plan.params)
        logger.info("기록 질문 계획: 규칙으로 정하지 못해 모델 계획으로 넘김 run=%s", request.run_id)
        return super()._plan(request)
