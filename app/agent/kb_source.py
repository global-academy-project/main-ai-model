"""KB 근거(citations[kind="KB"])에 붙이는 출처 정보 (C 요청 11/05: sourceType · kbCollection · docId).

서버는 KB 근거를 근거 테이블(evidence_citations)에 저장할 때 이 셋이 있어야 한다. 없거나 값이 맞지 않으면
output_json에만 남기고 건너뛴다. 그래서 확실한 것만 채우고, 하나라도 확인되지 않으면 셋 다 비운다 (추측하지 않는다).

  docId         검색 결과 hits[].doc_id 그대로 ("{source_type}:{kb_source_id}")
  sourceType    doc_id 앞부분. 아래 KB_SOURCE_TYPES에 있을 때만
  kbCollection  검색한 도구 기준 (search_accidents → kb_accidents, search_guides → kb_guides)

검색 도구 출력({"hits": [{"id", "doc_id", "source", "text", "score"}]})에는 source_type이 따로 없으므로 doc_id 앞부분을 쓴다.

끄는 법: ATTACH_KB_SOURCE = False → 항상 빈 dict (Citation의 세 필드는 None).
"""

from __future__ import annotations

from .evidence_collector.vocab import DEFAULT_CORPORA

ATTACH_KB_SOURCE = True

# C가 서버에서 받는 출처 종류 (11/05 요청서). A의 적재 코드(ingest_main.COLLECTIONS)와 같은지 확인 필요.
KB_SOURCE_TYPES: frozenset[str] = frozenset(
    {"ACCIDENT_REPORT", "SIF", "NEAR_MISS", "KOSHA_GUIDE", "RISK_GUIDE", "CHECKLIST", "LAW"}
)
# 검색 도구 → 컬렉션 (C 요청서 표)
TOOL_COLLECTIONS: dict[str, str] = {"search_accidents": "kb_accidents", "search_guides": "kb_guides"}


def collection_for(corpus: object) -> str | None:
    """자료 묶음 이름(EvidenceItem.corpus: "accident"·"guide")의 검색 도구로 컬렉션을 정한다. 모르면 None."""
    if not isinstance(corpus, str):
        return None
    entry = DEFAULT_CORPORA.get(corpus)
    return TOOL_COLLECTIONS.get(entry.tool) if entry is not None else None


def kb_citation_fields(doc_id: object, corpus: object) -> dict[str, str]:
    """Citation(kind="KB", ...)에 `**`로 펼쳐 넣을 값. 확인되지 않으면 빈 dict."""
    if not ATTACH_KB_SOURCE or not isinstance(doc_id, str):
        return {}
    source_type, sep, source_id = doc_id.partition(":")
    if not sep or source_type not in KB_SOURCE_TYPES or not source_id or source_id != source_id.strip():
        return {}
    collection = collection_for(corpus)
    if collection is None:
        return {}
    return {"source_type": source_type, "kb_collection": collection, "doc_id": doc_id}
