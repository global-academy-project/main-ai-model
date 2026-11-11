from app.agent.evidence_collector.models import SearchQuery
from app.agent.evidence_collector.recorder import StepEmitter
from app.agent.evidence_collector.searcher import Searcher, SearchState
from app.agent.evidence_collector.vocab import Corpus, expand_search_query


class CapturingTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((name, arguments))
        return {"hits": []}


def test_expand_search_query_uses_known_work_type_aliases() -> None:
    assert expand_search_query("화기작업 화재 사고") == "화기작업 화재 사고 용접"
    assert expand_search_query("용접 화재 사고") == "용접 화재 사고 화기"
    assert expand_search_query("산소 농도 기준") == "산소 농도 기준"


def test_searcher_sends_expanded_query_and_records_both_forms() -> None:
    tools = CapturingTools()
    records = []
    searcher = Searcher(
        tools=tools,
        corpora={"accident": Corpus("사고사례", "search_accidents")},
        k=5,
        min_score=None,
        emitter=StepEmitter(records.append),
    )
    original = "화기작업 화재 사고"

    searcher.run_round(
        [SearchQuery("accident", original, "유사 사고")],
        1,
        SearchState(),
    )

    assert tools.calls[0] == (
        "search_accidents",
        {"query": "화기작업 화재 사고 용접", "k": 5},
    )
    recorded_query = records[0].input["queries"][0]
    assert recorded_query["query"] == original
    assert recorded_query["expanded_query"] == "화기작업 화재 사고 용접"