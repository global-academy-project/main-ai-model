"""EvidenceCollector: 계획 → 검색 → 충분성 → 재질의를 이어 EvidencePack을 만든다.

단계별 일은 planner / searcher / sufficiency가 하고, 여기서는 순서와 멈추는 조건만 정한다.
엔진이 아니다. LLMClient(엔진)와 ToolInvoker(도구 묶음)를 받아서 쓰는 쪽이다. 설정은 읽지 않고 생성자로 받는다.

실패 정책:
  - LLM 호출 실패(LLMError)는 그대로 올린다. 루프 러너가 to_error_info로 agent_runs에 기록한다.
  - 모든 검색이 실패하면 ToolError(EVIDENCE_SEARCH_FAILED).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from app.llm.llm_base import LLMClient
from app.tools.tool_contract import ToolInvoker
from app.exceptions import ToolError
from . import sufficiency
from .models import (
    EvidencePack,
    RequirementStatus,
    SearchQuery,
    StepRecorder,
    WorkContext,
)
from .planner import Plan, Planner
from .recorder import StepEmitter
from .searcher import Searcher, SearchState
from .vocab import (
    DEFAULT_CORPORA,
    DEFAULT_REQUIREMENTS,
    WORK_TYPES,
    Corpus,
    EvidenceRequirement,
)


def _validate_config(
    corpora: Mapping[str, Corpus],
    requirements: Mapping[str, Sequence[EvidenceRequirement]],
    k: int,
    max_requery_rounds: int,
) -> None:
    if not corpora:
        raise ValueError("corpora가 비어 있습니다.")
    if not 1 <= k <= 20:
        raise ValueError("k는 1~20이어야 합니다.")
    if not 0 <= max_requery_rounds <= 5:
        raise ValueError("max_requery_rounds는 0~5여야 합니다.")
    for name, corpus in corpora.items():
        if not isinstance(corpus.tool, str) or not corpus.tool.strip():
            raise ValueError(f"corpus {name!r}의 tool 이름이 비어 있습니다.")
    for work_type, reqs in requirements.items():
        if work_type not in WORK_TYPES:
            raise ValueError(f"requirements에 모르는 작업유형: {work_type!r}")
        for req in reqs:
            if req.corpus not in corpora:
                raise ValueError(f"필수 근거 {req.key!r}의 corpus {req.corpus!r}가 corpora에 없습니다.")


class EvidenceCollector:
    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolInvoker,
        corpora: Mapping[str, Corpus] | None = None,
        requirements: Mapping[str, Sequence[EvidenceRequirement]] | None = None,
        k: int = 5,
        min_score: float | None = None,
        max_requery_rounds: int = 3,
        max_tokens: int = 800,
        on_step: StepRecorder | None = None,
    ) -> None:
        corpora = dict(corpora if corpora is not None else DEFAULT_CORPORA)
        self._requirements = dict(requirements if requirements is not None else DEFAULT_REQUIREMENTS)
        _validate_config(corpora, self._requirements, k, max_requery_rounds)
        self._max_requery_rounds = max_requery_rounds
        self._emitter = StepEmitter(on_step)
        self._planner = Planner(llm=llm, corpora=corpora, max_tokens=max_tokens)
        self._searcher = Searcher(tools=tools, corpora=corpora, k=k, min_score=min_score, emitter=self._emitter)

    def collect(self, work: WorkContext) -> EvidencePack:
        started = time.perf_counter()
        plan = self._planner.plan(work)
        self._emitter.emit("plan", {"work": work.model_dump()}, plan.dump(), started)

        reqs = sufficiency.requirements_for(self._requirements, plan.work_types)
        state = SearchState()
        statuses, rounds = self._search_until_enough(plan, reqs, state)

        if state.all_failed:
            raise ToolError("EVIDENCE_SEARCH_FAILED", "근거 검색이 모두 실패했습니다.")

        return EvidencePack(
            summary=plan.summary,
            work_types=plan.work_types,
            materials=plan.materials,
            equipment=plan.equipment,
            plan_source=plan.source,
            items=state.ranked_items(),
            requirements=statuses,
            missing=[s.description for s in statuses if not s.satisfied],
            search_failures=state.failures,
            rounds=rounds,
        )

    def _search_until_enough(
        self, plan: Plan, reqs: sufficiency.RequirementList, state: SearchState
    ) -> tuple[list[RequirementStatus], int]:
        """첫 검색 뒤 빠진 항목만 질의를 바꿔 다시 찾는다. (충족 상태, 실제로 돈 회차 수)."""
        round_no = 1
        self._searcher.run_round(plan.queries, round_no, state)
        statuses = sufficiency.check(reqs, state, round_no, self._emitter)

        for round_no in range(2, 2 + self._max_requery_rounds):
            missing = [req for (_, req), status in zip(reqs, statuses, strict=True) if not status.satisfied]
            if not missing:
                return statuses, round_no - 1
            queries = [
                SearchQuery(req.corpus, text, f"보강:{req.key}")
                for req in missing
                if (text := sufficiency.requery(req, plan, round_no)) is not None
            ]
            if not queries:  # 바꿔 볼 질의가 더 없다
                return statuses, round_no - 1
            new_items = self._searcher.run_round(queries, round_no, state)
            statuses = sufficiency.check(reqs, state, round_no, self._emitter)
            if new_items == 0:  # 질의를 바꿔도 새 근거가 없으면 더 돌아도 소용없다
                return statuses, round_no
        return statuses, round_no
