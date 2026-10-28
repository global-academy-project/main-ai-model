"""근거 수집기의 입출력 타입. WorkContext·EvidencePack은 HTTP로 주고받게 되면 app/schemas/로 옮긴다."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkContext(BaseModel):
    """등록 대화·되묻기 결과."""

    text: str = Field(min_length=1, max_length=4000, description="작업책임자가 쓴 작업 내용 원문")
    location: str | None = None
    hot_work_kind: str | None = Field(default=None, description="화기 종류 (용접·절단·그라인더 등)")
    nearby_materials: list[str] = Field(default_factory=list)
    headcount: int | None = Field(default=None, ge=1)
    end_time: str | None = None
    equipment: list[str] = Field(default_factory=list)
    work_types: list[str] = Field(default_factory=list, description="classify_work 결과가 있으면 넣는다. 계획 결과와 합친다")


class EvidenceItem(BaseModel):
    id: str  # RAG 청크 ID
    doc_id: str
    source: str  # 인용 표시
    text: str
    score: float
    corpus: str
    round: int  # 몇 번째 검색 회차에서 처음 찾았나 (1 = 첫 검색)
    matched_by: list[str] = Field(default_factory=list)  # 이 근거를 찾은 질의의 목적들


class RequirementStatus(BaseModel):
    key: str
    work_type: str
    description: str
    satisfied: bool
    evidence_ids: list[str] = Field(default_factory=list)


class SearchFailure(BaseModel):
    round: int
    corpus: str
    query: str
    code: str  # ToolError.code
    reason: str  # ToolError.message (도구가 직접 쓴 고정 문구)


class ToolHit(BaseModel):
    """검색 도구 출력의 근거 하나. 모르는 필드는 무시한다."""

    id: str
    doc_id: str
    source: str
    text: str
    score: float


class SearchToolOutput(BaseModel):
    hits: list[ToolHit] = Field(default_factory=list)


class EvidencePack(BaseModel):
    summary: str
    work_types: list[str]
    materials: list[str]  # lookup_msds에 넘길 물질
    equipment: list[str]
    plan_source: Literal["llm", "fallback"]
    items: list[EvidenceItem]  # 점수 높은 순
    requirements: list[RequirementStatus]
    missing: list[str]  # 끝내 못 찾은 필수 근거 설명. 판단 단계는 이 항목을 근거 없이 단정하면 안 된다
    search_failures: list[SearchFailure]
    rounds: int  # 실제로 돈 검색 회차 수


@dataclass(frozen=True)
class SearchQuery:
    corpus: str
    query: str
    purpose: str


@dataclass(frozen=True)
class StepRecord:
    """agent_steps 기록용. 이 모듈은 DB를 모르고, 루프 러너가 on_step으로 받아 저장한다."""

    name: Literal["plan", "search", "sufficiency"]
    input: dict[str, Any]
    output: dict[str, Any]
    elapsed_ms: int


StepRecorder = Callable[[StepRecord], None]
