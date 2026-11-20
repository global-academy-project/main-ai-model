"""기록 질문 의존성 (규칙 우선 계획판, B). `app/main.py`가 RECORD_PLAN_RULES가 켜졌을 때만
D의 `get_record_query_agent`를 이 함수로 바꿔 끼운다 (`app.dependency_overrides`). D의 파일은 고치지 않는다.

설정 검사·내부 API 연결은 D의 함수를 그대로 불러 같은 결과를 얻고, 같은 모델·같은 검색 도구로 규칙 우선 에이전트를 만든다.
D가 `RecordQueryAgent`의 `_llm`·`_search_records` 이름을 바꾸면 시험(`test_record_plan_rules`)이 알려 준다.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.agent.record_query import RecordQueryAgent
from app.agent.record_query_rules_agent import RuleFirstRecordQueryAgent
from app.api.v1.agent.record_query_deps import RecordQueryLLM, get_record_query_agent
from app.core import Settings, get_settings


def get_rule_first_record_query_agent(
    llm: RecordQueryLLM,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RecordQueryAgent:
    base = get_record_query_agent(llm, settings)  # D의 설정 검사와 연결 그대로
    return RuleFirstRecordQueryAgent(llm=base._llm, search_records=base._search_records)
