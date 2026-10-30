from __future__ import annotations
from pydantic import Field
from app.schemas import CamelModel
from app.schemas.forms import PermitFormType

class Classification(CamelModel):
    work_types: list[str]
    high_risk: bool | None = Field(description="작업유형을 못 읽으면 null(모름)")
    required_permits: list[PermitFormType]
    can_start_before_approval: bool
    reasons: list[ClassificationReason] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list, description="부정돼서 뺀 표현 (예: '크레인 안 씀')")
    out_of_scope: list[str] = Field(default_factory=list, description="고위험이지만 1개월 범위 밖이라 허가서를 만들지 않은 작업유형")

class ClassificationReason(CamelModel):
    work_type: str
    keyword: str

