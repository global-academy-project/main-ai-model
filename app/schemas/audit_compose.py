"""감사 대응 보고서 구성기(B) 요청·응답 계약.

구성기는 LLM 없이 코드만으로 묶음·상태값·서술문·종합 내역을 만든다.
요청 본문은 기존 `/audit-report` 본문의 상위 집합이라(`total`·`truncated` 추가) Spring이
같은 본문을 양쪽에 보낼 수 있다. 이 파일은 B가 소유하고, D·C는 읽기만 한다.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .base_schemas import CamelModel

# 항목 상태. 화면·인쇄에는 한글 라벨(`state_label`)만 보인다.
ItemState = Literal["REVIEW", "REFERENCE", "OK", "UNAVAILABLE"]
# 확인 불가 사유. NOT_SUPPORTED = 조회 항목 자체가 아직 없음, NO_DATA = 입력에 없음, INVALID = 값이 이상함
UnavailableReason = Literal["NOT_SUPPORTED", "NO_DATA", "INVALID"]
BasisKind = Literal["LEGAL", "INTERNAL"]


# ---------------------------------------------------------------- 요청


class AuditComposeItem(CamelModel):
    """Spring 집계 항목 하나. 모르는 필드는 무시한다(기존 `/audit-report` 본문과 호환)."""

    query_type: str = Field(min_length=1)
    title: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
    total: int | None = Field(default=None, strict=True, ge=0)
    truncated: bool | None = None
    records: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class AuditComposeInput(CamelModel):
    report_id: int = Field(strict=True, gt=0)
    site_id: int = Field(strict=True, gt=0)
    date_from: str = Field(min_length=10, max_length=10)
    date_to: str = Field(min_length=10, max_length=10)
    items: list[AuditComposeItem] = Field(min_length=1)

    @field_validator("date_from", "date_to")
    @classmethod
    def valid_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.") from exc
        return value

    @model_validator(mode="after")
    def valid_period_and_items(self) -> "AuditComposeInput":
        if self.date_from > self.date_to:
            raise ValueError("조회 시작일은 종료일보다 늦을 수 없습니다.")
        types = [item.query_type for item in self.items]
        if len(types) != len(set(types)):
            raise ValueError("감사보고서 조회 항목은 중복될 수 없습니다.")
        return self


# ---------------------------------------------------------------- 응답


class ComposeMeta(CamelModel):
    report_id: int
    site_id: int
    date_from: str
    date_to: str
    generated_at: str


class ComposeScope(CamelModel):
    """점검 범위 건수. 구할 수 없는 값은 None(화면에서는 '확인 불가')."""

    works: int | None = None
    permits: int | None = None
    near_misses: int | None = None
    corrective_actions: int | None = None


class OverviewPoint(CamelModel):
    area: str
    point: str
    kind: BasisKind


class OverviewGroup(CamelModel):
    no: int
    area: str
    items: list[OverviewPoint] = Field(default_factory=list)


class ComposeOverview(CamelModel):
    scope: ComposeScope
    groups: list[OverviewGroup] = Field(default_factory=list)


class SummaryRow(CamelModel):
    area: str
    label: str
    checked: int | None = None
    count: int | None = None
    rate_pct: float | None = None
    state: ItemState
    state_label: str


class SummaryGroup(CamelModel):
    no: int
    area: str
    rows: list[SummaryRow] = Field(default_factory=list)


class ComposeSummary(CamelModel):
    groups: list[SummaryGroup] = Field(default_factory=list)


class ItemBasis(CamelModel):
    title: str
    legal_basis: str | None = None
    kind: BasisKind


class FactValue(CamelModel):
    label: str
    value: float | int
    unit: str = "건"


class ItemFacts(CamelModel):
    denominator: int | None = None
    numerator: int | None = None
    rate_pct: float | None = None
    total: int | None = None
    shown: int = 0
    truncated: bool = False
    values: list[FactValue] = Field(default_factory=list)


class EvidenceRef(CamelModel):
    type: str
    id: int


class ComposeSentence(CamelModel):
    text: str
    source: Literal["TEMPLATE", "LLM"] = "TEMPLATE"
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ItemRecords(CamelModel):
    lines: list[str] = Field(default_factory=list)
    shown: int = 0
    total: int | None = None


class ActionStatus(CamelModel):
    open: int
    completed: int
    verified: int
    overdue: int


class ComposeItem(CamelModel):
    seq: int
    query_type: str
    area: str
    kind: BasisKind
    state: ItemState
    state_label: str
    unavailable_reason: UnavailableReason | None = None
    basis: list[ItemBasis] = Field(default_factory=list)
    facts: ItemFacts = Field(default_factory=ItemFacts)
    sentences: list[ComposeSentence] = Field(default_factory=list)
    records: ItemRecords = Field(default_factory=ItemRecords)
    action_status: ActionStatus | None = None


class ComposeGroup(CamelModel):
    doc_no: str
    seq: int
    area: str
    items: list[ComposeItem] = Field(default_factory=list)
    # 사람이 입력할 때까지 null. 구성기는 채우지 않는다.
    responder_opinion: str | None = None
    requested_action: str | None = None


class ReviewNeeded(CamelModel):
    area: str
    count: int
    record_ids: list[int] = Field(default_factory=list)
    # 화면에 보일 기록 번호(허가 번호). record_ids는 내부 id라 화면에 보이지 않는다.
    record_labels: list[str] = Field(default_factory=list)
    text: str


class OverallLine(CamelModel):
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ComposeOverall(CamelModel):
    counts: ComposeScope
    lines: list[str] = Field(default_factory=list)
    review_needed: list[ReviewNeeded] = Field(default_factory=list)
    links: list[OverallLine] = Field(default_factory=list)
    reference: list[OverallLine] = Field(default_factory=list)
    human_tasks: list[str] = Field(default_factory=list)
    # 종합 의견 칸에 미리 넣을 초안. lines·확인 필요·현황 참고·항목 간 연결의 문장을 줄바꿈으로 이은 것이다
    # (human_tasks는 넣지 않는다). 새 문장을 만들지 않고 위 목록의 문장만 쓴다. 끄면 빈 문자열.
    draft_text: str = ""


class ComposeVerification(CamelModel):
    code: str
    level: Literal["INFO", "WARN"]
    message: str


class AuditComposeResult(CamelModel):
    form_type: Literal["AUDIT_REPORT"] = "AUDIT_REPORT"
    schema_version: Literal["v1"] = "v1"
    # INCOMPLETE = 조회 실패·입력 이상으로 확인 불가가 된 항목이 있음(조회 항목 미지원은 제외)
    status: Literal["COMPLETE", "INCOMPLETE"]
    meta: ComposeMeta
    overview: ComposeOverview
    summary: ComposeSummary
    groups: list[ComposeGroup] = Field(default_factory=list)
    overall: ComposeOverall
    verification: list[ComposeVerification] = Field(default_factory=list)
