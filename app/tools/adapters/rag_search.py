"""A의 RAG 검색을 검색 도구 규격(지침 6.4)으로 감싸는 어댑터.

    도구 입력  {"query": str, "k": int, "min_score": float(선택), **그 밖의 키는 backend의 filters로 전달}
    도구 출력  {"hits": [{"id", "doc_id", "source", "text", "score"}]}  점수 높은 순, 최대 k개

이 모듈은 RAG 라이브러리를 임포트하지 않는다. A의 검색 함수(backend)를 인자로 받아 감싸기만 하므로
`app.rag`가 어떤 모양이든 backend 함수 하나만 맞추면 된다.

    backend(query: str, *, k: int, filters: Mapping[str, Any]) -> Sequence[hit]
    hit  = dict 또는 객체. 필드 id · doc_id · source · text · score

규칙
  - 점수가 낮아도 상위 k개를 돌려준다. `min_score`를 줬을 때만 거른다 (6.4).
  - 도구에 고정된 filters(예: 사고사례 전용)는 호출자가 넘긴 같은 이름의 키보다 우선한다 (자료 종류를 못 넘나들게).
  - `backend_errors`로 지정한 예외만 `ToolError("TOOL_UNAVAILABLE")`로 바꾼다. 그 밖의 예외(프로그래밍 오류)는 그대로 올린다.
    예외 원문은 응답에 싣지 않고 로그에만 남긴다 (고정 문구, 지침 4장).
  - 청크 `id`는 uuid여야 한다(`evidence_citations.kb_point_id`에 들어감). 아니면 `TOOL_OUTPUT_INVALID`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.exceptions import ToolError

logger = logging.getLogger(__name__)

DEFAULT_K = 5
MAX_K = 50
_RESERVED_KEYS = frozenset({"query", "k", "min_score"})
_HIT_FIELDS = ("id", "doc_id", "source", "text", "score")

SearchBackend = Callable[..., Sequence[Any]]
ToolFunction = Callable[[Mapping[str, Any]], dict[str, Any]]

SEARCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "검색 문장"},
        "k": {"type": "integer", "description": "돌려받을 최대 개수", "minimum": 1, "maximum": MAX_K},
        "min_score": {"type": "number", "description": "이 점수 미만은 거른다 (생략하면 거르지 않음)"},
    },
    "required": ["query"],
}


def make_search_tool(
    tool_name: str,
    backend: SearchBackend,
    *,
    filters: Mapping[str, Any] | None = None,
    backend_errors: tuple[type[Exception], ...] = (),
    require_uuid: bool = True,
) -> ToolFunction:
    """`tool_name` 도구 함수를 만든다. `filters`는 이 도구에 고정된 검색 조건(자료 종류 등)이다."""
    fixed_filters = dict(filters or {})

    def tool(arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("TOOL_INPUT_INVALID", f"{tool_name}에는 검색어(query)가 필요합니다.", tool=tool_name)
        k = _read_k(arguments.get("k", DEFAULT_K), tool_name)
        min_score = _read_min_score(arguments.get("min_score"), tool_name)
        call_filters = {**{key: v for key, v in arguments.items() if key not in _RESERVED_KEYS}, **fixed_filters}

        try:
            raw_hits = backend(query.strip(), k=k, filters=call_filters)
        except backend_errors as exc:
            logger.warning("검색 실패: tool=%s", tool_name, exc_info=exc)
            raise ToolError("TOOL_UNAVAILABLE", f"{tool_name} 검색에 실패했습니다.", tool=tool_name) from exc

        hits = [_to_hit(h, tool_name, require_uuid) for h in raw_hits]
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= min_score]
        hits.sort(key=lambda h: -h["score"])  # 정렬은 안정적이라 같은 점수는 backend 순서를 유지한다
        return {"hits": hits[:k]}

    return tool


def _read_k(value: Any, tool_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_K:
        raise ToolError("TOOL_INPUT_INVALID", f"{tool_name}의 k는 1~{MAX_K}의 정수여야 합니다.", tool=tool_name)
    return value


def _read_min_score(value: Any, tool_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("TOOL_INPUT_INVALID", f"{tool_name}의 min_score는 숫자여야 합니다.", tool=tool_name)
    return float(value)


def _field(hit: Any, name: str) -> Any:
    return hit.get(name) if isinstance(hit, Mapping) else getattr(hit, name, None)


def _to_hit(hit: Any, tool_name: str, require_uuid: bool) -> dict[str, Any]:
    values = {name: _field(hit, name) for name in _HIT_FIELDS}
    if any(v is None for v in values.values()):
        missing = [n for n, v in values.items() if v is None]
        logger.warning("검색 결과에 없는 필드: tool=%s fields=%s", tool_name, missing)
        raise ToolError("TOOL_OUTPUT_INVALID", f"{tool_name} 결과가 규격과 다릅니다 (필드 누락).", tool=tool_name)
    try:
        score = float(values["score"])
        chunk_id = str(values["id"])
        if require_uuid:
            uuid.UUID(chunk_id)
    except (TypeError, ValueError) as exc:
        raise ToolError("TOOL_OUTPUT_INVALID", f"{tool_name} 결과가 규격과 다릅니다 (id 또는 score).", tool=tool_name) from exc
    return {
        "id": chunk_id,
        "doc_id": str(values["doc_id"]),
        "source": str(values["source"]),
        "text": str(values["text"]),
        "score": score,
    }
