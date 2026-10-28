"""시나리오 분류용 참고 자료 검색: 사용자 말로 검색 도구(mock 포함)를 불러 상위 근거를 돌려준다.

- 분류 엔진(CATEGORIZATOR)은 도구 호출(tool use)을 지원하지 않는다. 그래서 **코드가** 도구를 부르고,
  결과만 분류 프롬프트에 붙인다 (`ScenarioRouter.route(references=...)`).
- 검색·합치기·실패 기록은 근거 수집기의 `Searcher`를 그대로 쓴다 (도구 규격도 같다).
- 참고 자료는 보조다. 도구 하나가 `ToolError`로 실패해도 나머지로 계속하고, 모두 실패해도 예외 없이
  빈 결과와 실패 기록을 돌려준다 (분류는 참고 자료 없이도 된다).
  `ToolError`가 아닌 예외(프로그래밍 오류)는 삼키지 않는다.
- 도구는 이름으로만 부른다 (`ToolInvoker`). mock인지 실제인지는 `tool_setup.py`가 정한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..evidence_collector.models import SearchFailure, SearchQuery
from ..evidence_collector.recorder import StepEmitter
from ..evidence_collector.searcher import Searcher, SearchState
from ..evidence_collector.textutil import one_line
from ..evidence_collector.vocab import DEFAULT_CORPORA, MAX_QUERY_CHARS, Corpus
from ..tool.tool_contract import ToolInvoker

PURPOSE = "시나리오 분류 참고"


@dataclass(frozen=True)
class Reference:
    id: str
    corpus: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    references: tuple[Reference, ...] = ()
    failures: tuple[SearchFailure, ...] = ()


class ReferenceRetriever:
    """엔진이 아니다. 도구 묶음을 받아 쓰는 쪽이다."""

    def __init__(
        self,
        tools: ToolInvoker,
        *,
        corpora: Mapping[str, Corpus] | None = None,
        k: int = 2,
        max_references: int = 3,
        min_score: float | None = None,
    ) -> None:
        corpora = dict(corpora if corpora is not None else DEFAULT_CORPORA)
        if not corpora:
            raise ValueError("corpora가 비어 있습니다.")
        if not 1 <= k <= 20:
            raise ValueError("k는 1~20이어야 합니다.")
        if max_references < 1:
            raise ValueError("max_references는 1 이상이어야 합니다.")
        self._tools = tools
        self._corpora = corpora
        self._k = k
        self._max_references = max_references
        self._min_score = min_score

    def retrieve(self, text: str) -> RetrievalResult:
        query = one_line(text or "", MAX_QUERY_CHARS)
        if not query:
            return RetrievalResult()
        searcher = Searcher(
            tools=self._tools,
            corpora=self._corpora,
            k=self._k,
            min_score=self._min_score,
            emitter=StepEmitter(None),
        )
        state = SearchState()
        searcher.run_round([SearchQuery(name, query, PURPOSE) for name in self._corpora], 1, state)
        items = state.ranked_items()[: self._max_references]
        return RetrievalResult(
            references=tuple(Reference(i.id, i.corpus, i.source, i.text, i.score) for i in items),
            failures=tuple(state.failures),
        )
