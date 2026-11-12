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
from .adapters.rag_tools import register_rag_tools
from .rag_loader import load_search_backend
from .tool_registry import ToolRegistry

CHECK_RULES_DESCRIPTION = "가스농도·필수 배치·필수 항목 판정 (Spring 규칙 엔진 호출)"

def build_tool_registry(
    *, extra: Callable[[ToolRegistry], None] | None = None, rag_backend: str | None = None,
    check_rules_tool: tuple[Callable, dict] | None = None,
) -> ToolRegistry:
    """B의 실제 도구 + A·C의 mock 도구. extra로 실제 도구를 덮어쓸 수 있다 (replace=True로 등록).

    rag_backend('모듈경로:함수이름', Settings.rag_backend)가 있으면 search_accidents · search_guides를 A의 검색으로 바꾼다.
    비어 있으면 mock 그대로다. 잘못된 값은 ValueError. 적용 순서는 mock → 실제 check_rules → RAG → extra다.

    check_rules_tool=(handler, parameters)가 있으면 check_rules를 C의 실제 도구로 바꾼다 (CHECK_RULES_TOOL=SPRING).
    만드는 곳은 app/agent/check_rules_setup.py다. None이면 mock 그대로다.
    """
    from .mock import register_mock_tools  # mock은 이 함수 안에서만 임포트한다

    registry = ToolRegistry()
    registry.register("classify_work", classify_work, owner="B", description="작업유형 판별, 고위험 여부와 필요한 허가 (규칙 기반)",
                      parameters=CLASSIFY_WORK_PARAMETERS)
    registry.register("fill_form", fill_form, owner="B", description="관리 항목 초안 작성과 스키마 검증", parameters=FILL_FORM_PARAMETERS)
    register_mock_tools(registry)
    if check_rules_tool is not None:  # C의 실제 check_rules (mock과 같은 입출력)
        handler, parameters = check_rules_tool
        registry.register("check_rules", handler, owner="C", description=CHECK_RULES_DESCRIPTION,
                          parameters=parameters, replace=True)
    if (rag := load_search_backend(rag_backend)) is not None:
        register_rag_tools(registry, rag.search, backend_errors=rag.errors)
    if extra is not None:
        extra(registry)
    return registry
