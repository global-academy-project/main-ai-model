from __future__ import annotations
from typing import Literal
from pydantic import Field, BaseModel
from app.schemas import CamelModel
from .request import WorkSlots
from .classification import Classification
from .evidence import EvidenceSummary, MsdsDto
from .rules import RuleCheckDto

class WorkRegisterResult(CamelModel):
    run_status: Literal["WAITING_INPUT", "DONE"]
    slots: WorkSlots
    ask: AskBack | None = None
    classification: Classification
    evidence: EvidenceSummary | None = None
    msds: list[MsdsDto] = Field(default_factory=list)
    rule_check: RuleCheckDto | None = None
    drafts: Drafts | None = None
    citations: list[Citation] = Field(default_factory=list, description="초안 전체에서 쓴 근거 (evidence_citations 저장용, 중복 제거)")
    approval_gate: ApprovalGate | None = None
    verification: list[VerificationIssue] = Field(default_factory=list)
    guidance: list[GuidanceNote] = Field(default_factory=list)
    llm_sources: LlmSources = Field(default_factory=LlmSources)
    steps: list[AgentStepDto] = Field(default_factory=list)

class AskBack(CamelModel):
    message: str
    missing_fields: list[MissingField]

class MissingField(BaseModel):
    key: str
    label: str
    question: str

class Drafts(CamelModel):
    risk_assessment: RiskAssessmentDraft
    permits: list[PermitDraft]
    inspection_checklist: InspectionChecklistDraft
    tbm: TbmDraft

class ApprovalGate(CamelModel):
    can_request: bool
    reasons: list[str]

class VerificationIssue(BaseModel):
    code: str
    message: str
    target: str | None = None

class GuidanceNote(BaseModel):
    title: str
    why: str

class LlmSources(BaseModel):
    plan: Literal["llm", "fallback"] | None = None
    judgment: Literal["llm", "fallback"] | None = None