from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.base_schemas import CamelModel


NearMissTagAxis = Literal["SEVERITY", "ACCIDENT_TYPE", "HAZARD", "WORK_TYPE", "SUBSTANCE"]


class NearMissTagRequest(CamelModel):
    run_id: int = Field(gt=0)
    site_id: int = Field(gt=0)
    near_miss_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=4000)
    immediate_action: str | None = Field(default=None, max_length=2000)
    zone_name: str | None = Field(default=None, max_length=300)
    work_type: str | None = Field(default=None, max_length=30)


class NearMissTagSuggestion(CamelModel):
    axis: NearMissTagAxis
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class NearMissTagResponse(CamelModel):
    tags: list[NearMissTagSuggestion] = Field(default_factory=list)
    tagging_source: str
    tagging_version: str
    model: str | None = None

