"""임시 패치: 검색 결과 doc_id가 없으면(벡터 파일 방식 적재) "출처 미상"으로 채워 통과시킨다.

A가 doc_id를 채우면 이 경로는 쓰이지 않는다. 그때까지만 필요한 시험이라 파일째 지워도 다른 시험에 영향이 없다.
외부 서비스는 쓰지 않는다(가짜 backend, 가짜 Qdrant 응답).
"""

import logging
import uuid

import pytest

from app.agent.kb_source import kb_citation_fields
from app.exceptions import ToolError
from app.rag.qdrant import _parse_hits
from app.tools.adapters import rag_search
from app.tools.adapters.rag_search import UNKNOWN_DOC_ID, make_search_tool
from app.tools.adapters.rag_tools import register_rag_tools
from app.tools.contract_check import check_search_output
from app.tools.tool_registry import ToolRegistry

POINT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "LAW:TEST:0"))


@pytest.fixture(autouse=True)
def reset_warning_flag(monkeypatch):
    monkeypatch.setattr(rag_search, "_warned_missing_doc_id", False)


def _hit(**changes):
    base = {"id": POINT_ID, "doc_id": None, "source": "제241조의2 (화재감시자)", "text": "본문", "score": 0.5}
    base.update(changes)
    return base


def _tool(hits, **kwargs):
    return make_search_tool("search_guides", lambda query, *, k, filters: hits, **kwargs)


def test_missing_doc_id_becomes_unknown_and_other_fields_are_kept():
    output = _tool([_hit()])({"query": "화재감시자"})
    assert output["hits"] == [
        {"id": POINT_ID, "doc_id": UNKNOWN_DOC_ID, "source": "제241조의2 (화재감시자)", "text": "본문", "score": 0.5}
    ]


def test_blank_doc_id_is_also_unknown():
    output = _tool([_hit(doc_id="  ")])({"query": "화재감시자"})
    assert output["hits"][0]["doc_id"] == UNKNOWN_DOC_ID


def test_given_doc_id_is_untouched():
    output = _tool([_hit(doc_id="LAW:12")])({"query": "화재감시자"})
    assert output["hits"][0]["doc_id"] == "LAW:12"


@pytest.mark.parametrize("field", ["id", "source", "text", "score"])
def test_other_missing_fields_still_fail(field):
    with pytest.raises(ToolError) as error:
        _tool([_hit(**{field: None})])({"query": "화재감시자"})
    assert error.value.code == "TOOL_OUTPUT_INVALID"


def test_switch_off_restores_the_old_behaviour(monkeypatch):
    monkeypatch.setattr(rag_search, "ALLOW_MISSING_DOC_ID", False)
    with pytest.raises(ToolError) as error:
        _tool([_hit()])({"query": "화재감시자"})
    assert error.value.code == "TOOL_OUTPUT_INVALID"


def test_output_passes_the_contract_check():
    output = _tool([_hit(), _hit(id=str(uuid.uuid4()), score=0.2)])({"query": "화재감시자", "k": 5})
    assert check_search_output(output, k=5) == []


def test_unknown_doc_id_leaves_the_kb_source_fields_empty():
    assert kb_citation_fields(UNKNOWN_DOC_ID, "guide") == {}
    assert kb_citation_fields(UNKNOWN_DOC_ID, "accident") == {}
    # A가 doc_id를 채우면 원래대로 채워진다.
    assert kb_citation_fields("LAW:12", "guide") == {
        "source_type": "LAW",
        "kb_collection": "kb_guides",
        "doc_id": "LAW:12",
    }


def test_warning_is_logged_once_per_process(caplog):
    tool = _tool([_hit(), _hit(id=str(uuid.uuid4()), score=0.2)])
    with caplog.at_level(logging.WARNING, logger=rag_search.logger.name):
        tool({"query": "화재감시자"})
        tool({"query": "화재감시자"})
    assert len([r for r in caplog.records if UNKNOWN_DOC_ID in r.getMessage()]) == 1


def test_real_qdrant_response_without_kb_source_id_passes_through_the_registry():
    """벡터 파일 방식으로 올린 점(payload에 kb_source_id 없음)이 실제 파서를 거쳐 도구까지 통과한다."""
    response = {
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
                    },
                }
            ]
        }
    }
    parsed = _parse_hits(response)
    assert parsed[0]["doc_id"] is None  # 백엔드는 아직 None을 돌려준다 (A 영역)

    registry = ToolRegistry()
    register_rag_tools(registry, lambda query, *, k, filters: parsed)
    for name in ("search_accidents", "search_guides"):
        hit = registry.call(name, {"query": "화재감시자"})["hits"][0]
        assert hit["doc_id"] == UNKNOWN_DOC_ID
        assert hit["source"] == "산업안전보건기준에 관한 규칙 (제241조의2)"
