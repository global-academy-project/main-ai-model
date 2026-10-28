"""관리 항목 JSON 스키마 (3일차 규격 ① 초안: 위험성평가 · 작업허가 · 점검표 · TBM).

계획서 3장 "모든 항목을 하나의 JSON 데이터로 만들고, 전산 등록 데이터와 인쇄용 양식 두 가지로 내보낸다"의 그 JSON이다.
에이전트는 이 모델로 초안을 만들고(fill_form), Spring은 같은 모양을 받아 업무 DB에 저장한다.

코드 값은 업무 DB 코드표(DB_Schemas_README 4장)와 같게 둔다.
  form_type        RISK_ASSESSMENT, HOT_WORK_PERMIT, CONFINED_SPACE_PERMIT, HEIGHT_WORK_PERMIT, INSPECTION, TBM
  assessment_type  DAILY
  측정 항목·단위    O2/PCT, LEL/PCT_LEL, H2S/PPM, CO/PPM, CO2/PCT
  배치 역할         FIRE_WATCHER, ATTENDANT, SUPERVISOR

확인 필요 (C, A):
  - 빈도 1~5 × 강도 1~4는 임시 척도다. risk_assessment_items의 CHECK 범위(V3)와 맞춰야 한다.
  - ACCIDENT_TYPES는 계획서 5장 예시를 늘린 임시 목록이다. tag_codes(V5)의 발생형태 코드와 맞춰야 한다.
"""

from __future__ import annotations
from typing import Literal
from pydantic import Field, computed_field, field_validator
from .base_schemas import CamelModel

# 5축 중 발생형태 · 위험요인 (계획서 5장). 발생형태는 임시 목록.
ACCIDENT_TYPES: tuple[str, ...] = (
    "화재", "폭발·파열", "누출·접촉", "산소결핍", "끼임", "떨어짐", "넘어짐",
    "부딪힘", "맞음", "깔림", "절단·베임", "감전", "기타",
)
RISK_FACTORS: tuple[str, ...] = ("기계적", "전기적", "화학적", "작업환경", "인적·관리")

FREQUENCY_MAX = 5
SEVERITY_MAX = 4

PermitFormType = Literal["HOT_WORK_PERMIT", "CONFINED_SPACE_PERMIT", "HEIGHT_WORK_PERMIT"]
FormType = Literal["RISK_ASSESSMENT", "HOT_WORK_PERMIT", "CONFINED_SPACE_PERMIT", "HEIGHT_WORK_PERMIT", "INSPECTION", "TBM"]
MeasurementItem = Literal["O2", "LEL", "H2S", "CO", "CO2"]
MeasurementUnit = Literal["PCT", "PCT_LEL", "PPM"]


class Citation(CamelModel):
    """근거 인용 하나. evidence_citations 테이블에 그대로 들어갈 수 있게 snippet을 필수로 둔다."""

    kind: Literal["KB", "MSDS", "RULE"]
    ref_id: str = Field(description="KB: 청크 ID(uuid5, kb_point_id) / MSDS: CAS 번호 또는 물질명 / RULE: 규칙 코드")
    source: str = Field(description="화면에 보일 출처 표시")
    snippet: str = Field(min_length=1, max_length=300, description="인용 원문 일부 (원본이 바뀌어도 근거가 남도록)")


class WorkPeriod(CamelModel):
    start: str | None = Field(default=None, description="HH:MM")
    end: str | None = Field(default=None, description="HH:MM")


class RiskItem(CamelModel):
    """위험성평가 위험요인 한 줄. 점수는 코드가 계산한다 (LLM이 곱셈하지 않는다)."""

    seq: int = Field(ge=1)
    hazard: str = Field(min_length=1, max_length=200, description="위험요인 (무엇이 어떻게 사고로 이어지나)")
    accident_types: list[str] = Field(min_length=1, description="발생형태 (5축)")
    risk_factor: str = Field(description="위험요인 분류 (5축)")
    frequency: int = Field(ge=1, le=FREQUENCY_MAX)
    severity: int = Field(ge=1, le=SEVERITY_MAX)
    measures: list[str] = Field(min_length=1, description="감소대책")
    residual_frequency: int = Field(ge=1, le=FREQUENCY_MAX)
    residual_severity: int = Field(ge=1, le=SEVERITY_MAX)
    citations: list[Citation] = Field(default_factory=list)
    needs_review: bool = Field(default=False, description="근거가 없거나 규칙으로 보충한 항목. 승인자가 꼭 봐야 한다")
    source: Literal["llm", "rule"]

    @field_validator("accident_types")
    @classmethod
    def _known_accident_types(cls, v: list[str]) -> list[str]:
        unknown = [a for a in v if a not in ACCIDENT_TYPES]
        if unknown:
            raise ValueError(f"모르는 발생형태: {unknown}")
        return v

    @field_validator("risk_factor")
    @classmethod
    def _known_risk_factor(cls, v: str) -> str:
        if v not in RISK_FACTORS:
            raise ValueError(f"모르는 위험요인 분류: {v}")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_score(self) -> int:
        return self.frequency * self.severity

    @computed_field  # type: ignore[prop-decorator]
    @property
    def residual_score(self) -> int:
        return self.residual_frequency * self.residual_severity


class RiskAssessmentDraft(CamelModel):
    form_type: Literal["RISK_ASSESSMENT"] = "RISK_ASSESSMENT"
    assessment_type: Literal["DAILY"] = "DAILY"
    title: str = "일일 위험성평가(JSA)"
    work_summary: str
    location: str | None
    work_types: list[str]
    period: WorkPeriod
    headcount: int | None = Field(default=None, ge=1)
    items: list[RiskItem] = Field(min_length=1)
    status: Literal["DRAFT"] = "DRAFT"


class MeasurementRequirement(CamelModel):
    item: MeasurementItem
    unit: MeasurementUnit
    value: float | None = None
    status: Literal["MISSING", "PASS", "FAIL"]
    criterion: str
    rule_code: str


class AssignmentRequirement(CamelModel):
    role: Literal["FIRE_WATCHER", "ATTENDANT", "SUPERVISOR"]
    label: str
    assignee: str | None = None
    status: Literal["ASSIGNED", "REQUIRED"]


class HazardousMaterial(CamelModel):
    name: str
    location: str | None = None
    category: str | None = Field(default=None, description="예: 인화성 액체 (MSDS GHS 분류)")
    cas_no: str | None = None


class PermitDraft(CamelModel):
    """작업허가서 초안 (화기 · 밀폐공간 · 고소). 허가 종류별로 쓰지 않는 칸은 비워 둔다."""

    form_type: PermitFormType
    title: str
    linked_permit_no: str | None = Field(default=None, description="연계 일반작업허가 번호. 신청 후 Spring이 채운다")
    location: str | None
    period: WorkPeriod
    headcount: int | None = Field(default=None, ge=1)
    hot_work_kind: str | None = None
    nearby_hazardous_materials: list[HazardousMaterial] = Field(default_factory=list)
    measurements: list[MeasurementRequirement] = Field(default_factory=list)
    assignments: list[AssignmentRequirement] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list, description="허가 조건으로 거는 안전조치")
    risk_item_seqs: list[int] = Field(default_factory=list, description="위험성평가와 같은 위험요인 (RiskItem.seq)")
    approval_blocked: bool
    block_reasons: list[str] = Field(default_factory=list)
    status: Literal["DRAFT"] = "DRAFT"


class ChecklistItem(CamelModel):
    seq: int = Field(ge=1)
    item: str
    target: str = Field(description="점검 대상 (장비 이름 또는 '작업구역')")
    reason: str = Field(description="이 항목을 넣은 이유 (투입 장비 · 작업유형)")


class InspectionChecklistDraft(CamelModel):
    form_type: Literal["INSPECTION"] = "INSPECTION"
    title: str = "장비·작업구역 일일 점검표"
    location: str | None
    items: list[ChecklistItem] = Field(min_length=1)
    status: Literal["DRAFT"] = "DRAFT"


class TbmPoint(CamelModel):
    seq: int = Field(ge=1)
    point: str
    risk_item_seqs: list[int] = Field(default_factory=list)


class TbmDraft(CamelModel):
    form_type: Literal["TBM"] = "TBM"
    title: str = "TBM 교육 포인트"
    points: list[TbmPoint] = Field(min_length=1)
    status: Literal["DRAFT"] = "DRAFT"


FORM_MODELS: dict[str, type[CamelModel]] = {
    "RISK_ASSESSMENT": RiskAssessmentDraft,
    "HOT_WORK_PERMIT": PermitDraft,
    "CONFINED_SPACE_PERMIT": PermitDraft,
    "HEIGHT_WORK_PERMIT": PermitDraft,
    "INSPECTION": InspectionChecklistDraft,
    "TBM": TbmDraft,
}
