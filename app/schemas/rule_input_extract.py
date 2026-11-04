"""문장 → 규칙 입력 LLM 추출 (Spring FastApiLlmExtractClient · LlmExtraction 과 같은 모양)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base_schemas import CamelModel

MeasurementItem = Literal["O2", "CO2", "LEL", "H2S", "CO", "NH3", "CL2"]
WorkType = Literal["HOT", "CONFINED", "HEIGHT"]
DEFAULT_ITEMS: list[MeasurementItem] = ["O2", "CO2", "LEL", "H2S", "CO", "NH3", "CL2"]
DEFAULT_WORK_TYPES: list[WorkType] = ["HOT", "CONFINED", "HEIGHT"]


class RuleInputExtractRequest(CamelModel):
    text: str = Field(min_length=1, max_length=2000, description="사용자가 입력한 문장")
    measurement_items: list[MeasurementItem] = Field(default_factory=lambda: list(DEFAULT_ITEMS))
    work_types: list[WorkType] = Field(default_factory=lambda: list(DEFAULT_WORK_TYPES))


class ExtractedMeasurement(CamelModel):
    item: str
    value: float = Field(description="원문에 적힌 숫자 그대로")
    unit: Literal["PCT", "PPM", "PCT_LEL"]
    evidence: str = Field(description="값이 나온 원문 부분 (원문 그대로)")


class ExtractedWorkType(CamelModel):
    type: str
    evidence: str


class RuleInputExtraction(CamelModel):
    measurements: list[ExtractedMeasurement] = Field(default_factory=list)
    work_types: list[ExtractedWorkType] = Field(default_factory=list)
    watcher_name: str | None = None
    watcher_evidence: str | None = None
