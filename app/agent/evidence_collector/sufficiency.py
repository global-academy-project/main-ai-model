"""③ 충분성 · ④ 재질의: 작업유형별 필수 근거 체크리스트와 비교하고, 빠진 항목의 다음 질의를 정한다.

1주차 범위: 충분성은 키워드 체크리스트로 판단한다 (되묻기를 필수 항목 체크리스트로 시작한 것과 같은 방식).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from .models import RequirementStatus
from .planner import Plan
from .recorder import StepEmitter
from .searcher import SearchState
from .textutil import one_line
from .vocab import MAX_QUERY_CHARS, EvidenceRequirement

RequirementList = list[tuple[str, EvidenceRequirement]]  # (작업유형, 필수 근거)


def requirements_for(
    requirements: Mapping[str, Sequence[EvidenceRequirement]], work_types: Sequence[str]
) -> RequirementList:
    """작업유형들의 필수 근거를 합친다 (같은 key는 한 번만)."""
    reqs: RequirementList = []
    seen: set[str] = set()
    for work_type in work_types:
        for req in requirements.get(work_type, ()):
            if req.key not in seen:
                seen.add(req.key)
                reqs.append((work_type, req))
    return reqs


def check(reqs: RequirementList, state: SearchState, round_no: int, emitter: StepEmitter) -> list[RequirementStatus]:
    started = time.perf_counter()
    ranked = state.ranked_items()
    statuses: list[RequirementStatus] = []
    for work_type, req in reqs:
        ids = [i.id for i in ranked if i.corpus == req.corpus and any(kw in i.text for kw in req.keywords)]
        statuses.append(
            RequirementStatus(
                key=req.key,
                work_type=work_type,
                description=req.description,
                satisfied=len(ids) >= req.min_hits,
                evidence_ids=ids,
            )
        )
    emitter.emit(
        "sufficiency",
        {"round": round_no, "requirements": [r.key for _, r in reqs]},
        {"missing": [s.key for s in statuses if not s.satisfied]},
        started,
    )
    return statuses


def requery(req: EvidenceRequirement, plan: Plan, round_no: int) -> str | None:
    """회차마다 다른 질의를 쓴다: 물질·장비를 붙임 → 작업 요약을 붙임 → 기본 문장만 (점점 넓게).
    물질·장비가 없으면 그 단계는 건너뛴다. 바꿔 볼 질의가 다 떨어지면 None."""
    focus = " ".join(plan.materials + plan.equipment)
    variants: list[str] = []
    for text in (f"{req.query} {focus}" if focus else "", f"{req.query} {plan.summary}", req.query):
        text = one_line(text, MAX_QUERY_CHARS)
        if text and text not in variants:
            variants.append(text)
    index = round_no - 2
    return variants[index] if index < len(variants) else None
