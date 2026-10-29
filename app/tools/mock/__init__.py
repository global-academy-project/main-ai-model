"""[연습용 mock] A·C 담당 도구의 자리 표시.

실제 모듈을 받으면 app/agent/tool_setup.py에서 같은 이름으로 등록하고 이 패키지는 쓰지 않는다.
에이전트 코드(work_register, evidence_collector)는 이 패키지를 임포트하지 않는다. 도구 이름으로만 부른다.
"""

from __future__ import annotations

from ..tool_registry import ToolRegistry
from .mock_msds import MSDS_PARAMETERS, lookup_msds
from .mock_rules import CHECK_RULES_PARAMETERS, check_rules
from .mock_search import SEARCH_PARAMETERS, search_accidents, search_guides

__all__ = ["register_mock_tools"]


def register_mock_tools(registry: ToolRegistry) -> None:
    registry.register("search_accidents", search_accidents, owner="A", mock=True,
                      description="[mock] 유사 사고·아차사고 검색 (재해조사보고서, SIF, 사내 아차사고)", parameters=SEARCH_PARAMETERS)
    registry.register("search_guides", search_guides, owner="A", mock=True,
                      description="[mock] 작업별 기준·대책·점검 항목 검색 (KOSHA GUIDE, 위험성평가 안내서)", parameters=SEARCH_PARAMETERS)
    registry.register("lookup_msds", lookup_msds, owner="A", mock=True,
                      description="[mock] 물질 유해성·응급조치 조회 (공단 MSDS API)", parameters=MSDS_PARAMETERS)
    registry.register("check_rules", check_rules, owner="C", mock=True,
                      description="[mock] 가스농도·필수 배치·필수 항목 판정 (규칙 엔진)", parameters=CHECK_RULES_PARAMETERS)
