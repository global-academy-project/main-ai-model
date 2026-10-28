"""도구 등록 한 곳. A·C·D의 실제 도구를 받으면 여기서만 바꾼다.

교체 방법 (예: A의 RAG 검색 모듈을 받았을 때)
    from app.<A의 패키지> import search_accidents, search_guides, lookup_msds
    registry.register("search_accidents", search_accidents, owner="A", description=..., parameters=..., replace=True)

교체 조건: 같은 도구 이름, 같은 입출력 규격 (지침 7.4. 검색 출력 {"hits": [{"id", "doc_id", "source", "text", "score"}]}).
에이전트 코드는 도구 이름으로만 부르므로 바꿀 곳은 이 파일뿐이다.
"""

from __future__ import annotations
from collections.abc import Callable
from app.agent.classify import CLASSIFY_WORK_PARAMETERS, classify_work
from app.agent.fill_form import FILL_FORM_PARAMETERS, fill_form
from .tool_registry import ToolRegistry

def build_tool_registry(*, extra: Callable[[ToolRegistry], None] | None = None) -> ToolRegistry:
    """B의 실제 도구 + A·C의 mock 도구. extra로 실제 도구를 덮어쓸 수 있다 (replace=True로 등록)."""
    from ..mock_tools import register_mock_tools  # mock은 이 함수 안에서만 임포트한다

    registry = ToolRegistry()
    registry.register("classify_work", classify_work, owner="B", description="작업유형 판별, 고위험 여부와 필요한 허가 (규칙 기반)",
                      parameters=CLASSIFY_WORK_PARAMETERS)
    registry.register("fill_form", fill_form, owner="B", description="관리 항목 초안 작성과 스키마 검증", parameters=FILL_FORM_PARAMETERS)
    register_mock_tools(registry)
    if extra is not None:
        extra(registry)
    return registry
