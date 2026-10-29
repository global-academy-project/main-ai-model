"""[연습용 mock] search_accidents · search_guides (A 담당 도구의 자리 표시).

규격은 evidence_collector가 쓰는 검색 도구 임시안(지침 7.4)을 그대로 따른다.
  입력  {"query": str, "k": int, "min_score": float(선택), ...그 밖의 키는 무시}
  출력  {"hits": [{"id", "doc_id", "source", "text", "score", "source_type", "tags"}]}   score 높은 순

검색 방식: 질의와 문서의 글자 2-gram 겹침 비율. 실제 벡터 검색처럼 점수가 낮아도 상위 k개를 돌려준다
(겹치는 단어가 없으면 빼는 가짜는 재질의 동작을 실제와 다르게 만든다 — 지침 8장).
청크 ID는 업무 DB evidence_citations.kb_point_id와 같은 uuid5(source_type:external_id:chunk_index) 형식이다.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from app.exceptions import ToolError
from .mock_corpus import ACCIDENT_DOCS, GUIDE_DOCS, MockDoc

# 실제 적재 코드(A)와 같은 네임스페이스를 쓰도록 맞춰야 한다. 지금은 임의 값.
MOCK_POINT_NAMESPACE = uuid.UUID("6f1c2b1e-6a3e-5c1d-9a57-3f0c1a2b9d10")
MAX_K = 20

SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "검색 문장"},
        "k": {"type": "integer", "minimum": 1, "maximum": MAX_K},
        "min_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["query", "k"],
}


def point_id(doc: MockDoc, chunk_index: int = 0) -> str:
    return str(uuid.uuid5(MOCK_POINT_NAMESPACE, f"{doc.source_type}:{doc.external_id}:{chunk_index}"))


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


class MockSearchTool:
    def __init__(self, name: str, docs: Sequence[MockDoc]) -> None:
        self.name = name
        self._docs = [(doc, point_id(doc), _bigrams(f"{doc.title} {doc.text}")) for doc in docs]

    def __call__(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query, k, min_score = arguments.get("query"), arguments.get("k", 5), arguments.get("min_score")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("TOOL_INPUT_INVALID", "검색 질의(query)가 비어 있습니다.", tool=self.name)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
            raise ToolError("TOOL_INPUT_INVALID", f"k는 1~{MAX_K} 사이의 정수여야 합니다.", tool=self.name)
        q = _bigrams(query)
        scored: list[tuple[float, str, MockDoc]] = []
        for doc, pid, grams in self._docs:
            score = len(q & grams) / len(q) if q else 0.0
            scored.append((round(score, 4), pid, doc))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if isinstance(min_score, (int, float)) and not isinstance(min_score, bool):
            scored = [t for t in scored if t[0] >= min_score]
        return {
            "hits": [
                {
                    "id": pid,
                    "doc_id": f"{doc.source_type}:{doc.external_id}",
                    "source": doc.title,
                    "text": doc.text,
                    "score": score,
                    "source_type": doc.source_type,
                    "tags": doc.tags,
                }
                for score, pid, doc in scored[:k]
            ]
        }


search_accidents = MockSearchTool("search_accidents", ACCIDENT_DOCS)
search_guides = MockSearchTool("search_guides", GUIDE_DOCS)
