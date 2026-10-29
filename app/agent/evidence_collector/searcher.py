"""② 검색: 질의마다 자료 종류별 검색 도구를 부르고, 같은 청크는 하나로 합친다.

검색 도구 규격 (A의 search_accidents, search_guides 등. 3일차 규격 ①로 확정하기 전의 임시안):
  입력  {"query": str, "k": int, "min_score": float(선택), **Corpus.arguments}
  출력  {"hits": [{"id", "doc_id", "source", "text", "score"}, ...]}   (id는 청크 ID, score 높은 순)
         출력에 다른 필드가 더 있어도 무시한다. 규격에 어긋나면 TOOL_OUTPUT_INVALID로 기록한다.
  실패  ToolError (app.agent.tool_contract)

ToolError는 search_failures에 남기고 계속한다. ToolError가 아닌 예외(프로그래밍 오류)는 그대로 올린다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel, ValidationError
from .models import EvidenceItem, SearchFailure, SearchQuery, SearchToolOutput, ToolHit
from .recorder import StepEmitter
from .vocab import Corpus
from app.exceptions import ToolError
from app.tools.tool_contract import ToolInvoker

logger = logging.getLogger(__name__)


@dataclass
class SearchState:
    """collect() 한 번 동안 쌓이는 검색 결과."""

    items: dict[str, EvidenceItem] = field(default_factory=dict)
    done: set[tuple[str, str]] = field(default_factory=set)
    failures: list[SearchFailure] = field(default_factory=list)
    attempts: int = 0
    successes: int = 0

    def ranked_items(self) -> list[EvidenceItem]:
        """점수 높은 순 (같으면 id 순)."""
        return sorted(self.items.values(), key=lambda i: (-i.score, i.id))

    @property
    def all_failed(self) -> bool:
        return bool(self.attempts) and self.successes == 0


class Searcher:
    def __init__(
        self,
        *,
        tools: ToolInvoker,
        corpora: Mapping[str, Corpus],
        k: int,
        min_score: float | None,
        emitter: StepEmitter,
    ) -> None:
        self._tools = tools
        self._corpora = corpora
        self._k = k
        self._min_score = min_score
        self._emitter = emitter

    def run_round(self, queries: Sequence[SearchQuery], round_no: int, state: SearchState) -> int:
        """질의를 모두 돌리고 새로 찾은 근거 수를 돌려준다."""
        started = time.perf_counter()
        new_items = 0
        ran: list[dict[str, str]] = []
        failures_before = len(state.failures)
        for q in queries:
            if (q.corpus, q.query) in state.done:
                continue
            state.done.add((q.corpus, q.query))
            state.attempts += 1
            corpus = self._corpora[q.corpus]
            ran.append({"corpus": q.corpus, "tool": corpus.tool, "query": q.query, "purpose": q.purpose})
            try:
                hits = self._call(corpus, q.query)
            except ToolError as exc:
                logger.warning(
                    "근거 검색 실패: round=%s corpus=%s tool=%s code=%s", round_no, q.corpus, corpus.tool, exc.code, exc_info=exc
                )
                state.failures.append(
                    SearchFailure(round=round_no, corpus=q.corpus, query=q.query, code=exc.code, reason=exc.message)
                )
                continue
            state.successes += 1
            new_items += self._absorb(hits, q, round_no, state)
        self._emitter.emit(
            "search",
            {"round": round_no, "queries": ran},
            {
                "new_items": new_items,
                "total_items": len(state.items),
                "failures": [f.model_dump() for f in state.failures[failures_before:]],
            },
            started,
        )
        return new_items

    def _call(self, corpus: Corpus, query: str) -> list[ToolHit]:
        arguments: dict[str, Any] = {**corpus.arguments, "query": query, "k": self._k}
        if self._min_score is not None:
            arguments["min_score"] = self._min_score
        raw = self._tools.call(corpus.tool, arguments)
        if isinstance(raw, BaseModel):
            raw = raw.model_dump()
        try:
            return SearchToolOutput.model_validate(raw).hits
        except ValidationError as exc:
            logger.warning("검색 도구 출력이 규격과 다릅니다: tool=%s", corpus.tool, exc_info=exc)
            raise ToolError(
                "TOOL_OUTPUT_INVALID", f"'{corpus.tool}' 도구의 출력이 검색 결과 규격과 다릅니다.", tool=corpus.tool
            ) from exc

    @staticmethod
    def _absorb(hits: Sequence[ToolHit], q: SearchQuery, round_no: int, state: SearchState) -> int:
        """근거를 합친다. 새 근거 수를 돌려준다."""
        new_items = 0
        for hit in hits:
            item = state.items.get(hit.id)
            if item is None:
                state.items[hit.id] = EvidenceItem(
                    id=hit.id,
                    doc_id=hit.doc_id,
                    source=hit.source,
                    text=hit.text,
                    score=hit.score,
                    corpus=q.corpus,
                    round=round_no,
                    matched_by=[q.purpose],
                )
                new_items += 1
            else:
                if q.purpose not in item.matched_by:
                    item.matched_by.append(q.purpose)
                item.score = max(item.score, hit.score)
        return new_items
