"""RAG 검색 도구 두 개(`search_accidents`, `search_guides`)를 레지스트리에 등록한다. mock을 실제로 바꾸는 한 줄 지점.

    register_rag_tools(registry, backend, backend_errors=(RagError,), filters_by_tool={...})

`replace=True`로 덮어쓰므로 mock이 먼저 깔려 있어야 한다. 이미 등록된 이름이 없으면 그냥 새로 등록된다.
`filters_by_tool`의 값(예: doc_type)은 A가 적재한 메타데이터와 맞춰야 한다. 아직 미정이라 기본값은 비어 있다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .rag_search import SEARCH_TOOL_PARAMETERS, SearchBackend, make_search_tool

SEARCH_TOOLS: dict[str, str] = {
    "search_accidents": "사고 사례를 검색한다 (재해 사례, 아차사고).",
    "search_guides": "안전 지침·기술지침·법령 조항을 검색한다.",
}


def register_rag_tools(
    registry: Any,
    backend: SearchBackend,
    *,
    backend_errors: tuple[type[Exception], ...] = (),
    filters_by_tool: Mapping[str, Mapping[str, Any]] | None = None,
    owner: str = "A",
) -> list[str]:
    """등록한 도구 이름을 돌려준다. `registry`는 `ToolRegistry`다 (여기서 임포트하지 않는다)."""
    filters_by_tool = filters_by_tool or {}
    registered: list[str] = []
    for name, description in SEARCH_TOOLS.items():
        function = make_search_tool(name, backend, filters=filters_by_tool.get(name), backend_errors=backend_errors)
        registry.register(name, function, owner=owner, description=description, parameters=SEARCH_TOOL_PARAMETERS, replace=True)
        registered.append(name)
    return registered
