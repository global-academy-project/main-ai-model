"""감사보고서 입력·출력 계약.

집계는 서버가 제공한다. 이 모듈은 집계 결과를 요약하고 수치를 검증하는 계약만 정의한다.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .base_schemas import CamelModel


AuditQueryType = Literal[
    "risk_assessment_rate",
    "permit_approval_statistics",
    "corrective_action_rate",
    "blocked_records",
    "missing_measurements",
]


class AuditReportItem(CamelModel):
    query_type: AuditQueryType
    title: str = Field(min_length=1)
    stats: dict[str, Any] = Field(default_factory=dict)
    records: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_stats(self) -> "AuditReportItem":
        if not self.stats:
            raise ValueError("감사보고서 집계(stats)는 비어 있을 수 없습니다.")
        return self


class AuditReportInput(CamelModel):
    report_id: int = Field(strict=True, gt=0)
    site_id: int = Field(strict=True, gt=0)
    date_from: str = Field(min_length=10, max_length=10)
    date_to: str = Field(min_length=10, max_length=10)
    items: list[AuditReportItem] = Field(min_length=1)
    data_snapshot: dict[str, Any] | None = None

    @field_validator("date_from", "date_to")
    @classmethod
    def valid_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.") from exc
        return value

    @model_validator(mode="after")
    def valid_period_and_items(self) -> "AuditReportInput":
        if self.date_from > self.date_to:
            raise ValueError("조회 시작일은 종료일보다 늦을 수 없습니다.")
        query_types = [item.query_type for item in self.items]
        if len(query_types) != len(set(query_types)):
            raise ValueError("감사보고서 조회 항목은 중복될 수 없습니다.")
        return self


class AuditVerification(CamelModel):
    status: Literal["MATCH", "MISMATCH"]
    expected_numbers: list[str] = Field(default_factory=list)
    actual_numbers: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)


class AuditReportDraftItem(CamelModel):
    query_type: AuditQueryType
    title: str
    summary: str
    verification: AuditVerification
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class AuditReportResult(CamelModel):
    report_id: int
    site_id: int
    date_from: str
    date_to: str
    status: Literal["DRAFT", "REVIEW_REQUIRED", "FAILED"]
    items: list[AuditReportDraftItem] = Field(default_factory=list)
    overall_verification: Literal["PASS", "FAIL"]
    data_snapshot: dict[str, Any] | None = None
