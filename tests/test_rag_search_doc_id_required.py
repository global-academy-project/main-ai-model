"""검색 어댑터: 검색 결과의 doc_id는 필수다 (임시 허용 T12를 걷어낸 뒤의 계약).

벡터 파일 payload에 kb_source_id·kb_chunk_id가 채워지므로 백엔드가 doc_id("LAW:12" 꼴)를 항상 돌려준다.
doc_id가 비면 그 검색은 TOOL_OUTPUT_INVALID 로 거부한다. 그래서 장부 등록 없이 올린 점이 섞이면 검색이 실패한다 (의도된 동작).
외부 서비스는 쓰지 않는다 (가짜 backend, 가짜 Qdrant 응답).
"""

import uuid

import pytest

from app.exceptions import ToolError
from app.tools.adapters import rag_search
from app.tools.adapters.rag_search import make_search_tool

POINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "LAW:TEST:0"))


def _hit(**changes):
    base = {"id": POINT_ID, "doc_id": "LAW:12", "source": "제241조의2 (화재감시자)", "text": "본문", "score": 0.5}
    base.update(changes)
    return base


def _tool(hits, **kwargs):
    return make_search_tool("search_guides", lambda query, *, k, filters: hits, **kwargs)


def test_given_doc_id_is_passed_through_untouched():
    output = _tool([_hit()])({"query": "화재감시자"})
    assert output["hits"] == [
        {"id": POINT_ID, "doc_id": "LAW:12", "source": "제241조의2 (화재감시자)", "text": "본문", "score": 0.5}
    ]


@pytest.mark.parametrize("field", ["id", "doc_id", "source", "text", "score"])
def test_any_missing_field_fails_including_doc_id(field):
    with pytest.raises(ToolError) as error:
        _tool([_hit(**{field: None})])({"query": "화재감시자"})
    assert error.value.code == "TOOL_OUTPUT_INVALID"


def test_one_unlinked_hit_fails_the_whole_search():
    hits = [_hit(), _hit(id=str(uuid.uuid4()), doc_id=None, score=0.2)]
    with pytest.raises(ToolError) as error:
        _tool(hits)({"query": "화재감시자"})
    assert error.value.code == "TOOL_OUTPUT_INVALID"


def test_the_temporary_fallback_is_gone():
    for name in ("ALLOW_MISSING_DOC_ID", "UNKNOWN_DOC_ID", "_warned_missing_doc_id", "_note_missing_doc_id"):
        assert not hasattr(rag_search, name), name


# ── 아래는 실제 app 모듈과 함께 도는 시험 (kb_source·qdrant 파서·레지스트리·계약 검사) ─────────────

def test_output_passes_the_contract_check():
    from app.tools.contract_check import check_search_output

    output = _tool([_hit(), _hit(id=str(uuid.uuid4()), score=0.2)])({"query": "화재감시자", "k": 5})
    assert check_search_output(output, k=5) == []


def test_linked_doc_id_fills_the_kb_source_fields():
    from app.agent.kb_source import kb_citation_fields

    assert kb_citation_fields("LAW:12", "guide") == {
        "source_type": "LAW",
        "kb_collection": "kb_guides",
        "doc_id": "LAW:12",
    }


def _qdrant_response(**payload_extra):
    return {
        "result": {
            "points": [
                {
                    "id": POINT_ID,
                    "score": 0.31,
                    "payload": {
                        "source_type": "LAW",
                        "title": "산업안전보건기준에 관한 규칙",
                        "locator": "제241조의2",
                        "text": "화재감시자를 지정해 ...",
                        **payload_extra,
                    },
                }
            ]
        }
    }


def test_real_qdrant_response_with_ledger_ids_passes_through_the_registry():
    """장부와 연결된 점(payload에 kb_source_id)이 실제 파서를 거쳐 두 검색 도구까지 통과한다."""
    from app.rag.qdrant import _parse_hits
    from app.tools.adapters.rag_tools import register_rag_tools
    from app.tools.tool_registry import ToolRegistry

    parsed = _parse_hits(_qdrant_response(kb_source_id=42, kb_chunk_id=7))
    assert parsed[0]["doc_id"] == "LAW:42"

    registry = ToolRegistry()
    register_rag_tools(registry, lambda query, *, k, filters: parsed)
    for name in ("search_accidents", "search_guides"):
        hit = registry.call(name, {"query": "화재감시자"})["hits"][0]
        assert hit["doc_id"] == "LAW:42"
        assert hit["source"] == "산업안전보건기준에 관한 규칙 (제241조의2)"


def test_real_qdrant_response_without_ledger_ids_is_rejected():
    """장부 id 없이 올린 점은 백엔드가 doc_id=None을 돌려주고, 어댑터가 거부한다 (T12 이전 동작)."""
    from app.rag.qdrant import _parse_hits

    parsed = _parse_hits(_qdrant_response())
    assert parsed[0]["doc_id"] is None
    with pytest.raises(ToolError) as error:
        _tool(parsed)({"query": "화재감시자"})
    assert error.value.code == "TOOL_OUTPUT_INVALID"
