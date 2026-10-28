"""작업 등록 에이전트 API 요청·응답 (POST /api/v1/agent/work-register).

FastAPI는 대화 상태를 저장하지 않는다(대화·실행 기록은 Spring의 agent_conversations · agent_runs가 갖는다).
그래서 되묻기 응답에 지금까지 모은 값(slots)을 돌려주고, 다음 요청에 그 slots를 그대로 다시 실어 보낸다.

    1차 요청  {"message": "오늘 오후에 용접 작업 있어"}
    1차 응답  runStatus=WAITING_INPUT, ask.missingFields=[장소, 화기 종류, ...], slots={...}
    2차 요청  {"message": "<답변>", "slots": <1차 응답의 slots>}
    2차 응답  runStatus=DONE, drafts={위험성평가, 화기작업허가서, 점검표, TBM}, steps=[도구 호출 기록]

runStatus 값은 agent_runs.status 코드(WAITING_INPUT, DONE)와 같다.
필드 표기는 base_schemas.CamelModel을 따라 camelCase로 내보내고, 요청은 snake_case·camelCase 둘 다 받는다
(Spring DTO 규칙이 정해지면 바꾼다. 지침 9장).
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import Field, BaseModel
from .base_schemas import CamelModel
from .forms import (
    Citation,
    InspectionChecklistDraft,
    MeasurementItem,
    PermitDraft,
    PermitFormType,
    RiskAssessmentDraft,
    TbmDraft,
)

MAX_UTTERANCES = 10


class WorkSlots(CamelModel):
    """되묻기 체크리스트 항목. 응답으로 받은 값을 다음 요청에 그대로 다시 보낸다.
    화면에서 직접 채운 값(예: 화재감시자 이름)도 여기에 넣어 보내면 된다."""

    utterances: list[str] = Field(default_factory=list, max_length=MAX_UTTERANCES, description="지금까지 받은 작업책임자 발화")
    location: str | None = None
    hot_work_kind: str | None = None
    nearby_materials: list[str] = Field(default_factory=list)
    nearby_materials_none: bool = Field(default=False, description="'주변 물질 없음'이라고 답했다")
    material_locations: dict[str, str] = Field(default_factory=dict, description="물질 → 보관 위치")
    headcount: int | None = Field(default=None, ge=1, le=500)
    start_time: str | None = None
    end_time: str | None = None
    equipment: list[str] = Field(default_factory=list)
    excluded_equipment: list[str] = Field(default_factory=list, description="'크레인 안 씀'처럼 쓰지 않는다고 한 장비")
    equipment_none: bool = False
    measurements: dict[MeasurementItem, float] = Field(default_factory=dict)
    fire_watcher: str | None = None
    attendant: str | None = None


class WorkRegisterRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000, description="이번 작업책임자 발화")
    slots: WorkSlots | None = Field(default=None, description="이전 응답의 slots. 첫 요청이면 비운다")
    mode: Literal["VERIFY", "GUIDE"] = Field(default="VERIFY", description="sites.mode. GUIDE면 쉬운 말 안내를 붙인다")


class MissingField(BaseModel):
    key: str
    label: str
    question: str


class AskBack(BaseModel):
    message: str
    missing_fields: list[MissingField]


class ClassificationReason(CamelModel):
    work_type: str
    keyword: str


class Classification(CamelModel):
    work_types: list[str]
    high_risk: bool | None = Field(description="작업유형을 못 읽으면 null(모름)")
    required_permits: list[PermitFormType]
    can_start_before_approval: bool
    reasons: list[ClassificationReason] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list, description="부정돼서 뺀 표현 (예: '크레인 안 씀')")
    out_of_scope: list[str] = Field(default_factory=list, description="고위험이지만 1개월 범위 밖이라 허가서를 만들지 않은 작업유형")


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


class RuleEvaluationDto(CamelModel):
    rule_code: str
    title: str
    status: Literal["PASS", "FAIL", "MISSING", "CHECK"]
    severity: Literal["BLOCK", "WARN"]
    message: str
    basis: str


class RuleCheckDto(BaseModel):
    blocked: bool
    evaluations: list[RuleEvaluationDto]
    conflicts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


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


class AgentStepDto(CamelModel):
    """agent_steps 한 줄. kind=TOOL이면 tool_name이 도구 이름, PHASE면 루프 단계(계획·충분성·판단·검증)."""

    seq: int
    kind: Literal["TOOL", "PHASE"]
    name: str
    tool_name: str | None
    status: Literal["OK", "ERROR"]
    input: dict[str, Any]
    output: dict[str, Any]
    elapsed_ms: int
    error_code: str | None = None


class LlmSources(BaseModel):
    plan: Literal["llm", "fallback"] | None = None
    judgment: Literal["llm", "fallback"] | None = None


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


class ToolInfo(BaseModel):
    name: str
    owner: str
    mock: bool
    description: str
