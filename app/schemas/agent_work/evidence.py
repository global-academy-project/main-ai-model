from __future__ import annotations
from typing import Literal
from pydantic import Field, BaseModel
from app.schemas import CamelModel

class EvidenceSummary(CamelModel):
    plan_source: Literal["llm", "fallback"]
    rounds: int
    items: list[EvidenceItemDto]
    requirements: list[RequirementDto]
    missing: list[str]
    search_failures: list[SearchFailureDto]

class MsdsDto(CamelModel):
    query: str
    found: bool
    name: str | None = None
    cas_no: str | None = None
    ghs_classes: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    source: str | None = None
    
class EvidenceItemDto(CamelModel):
    id: str
    doc_id: str
    corpus: str
    source: str
    snippet: str
    score: float
    round: int
    matched_by: list[str]

class RequirementDto(CamelModel):
    key: str
    work_type: str
    description: str
    satisfied: bool
    evidence_ids: list[str]

class SearchFailureDto(BaseModel):
    round: int
    corpus: str
    query: str
    code: str
    reason: str