from __future__ import annotations

from typing import Any

from pydantic import Field

from .base_schemas import CamelModel


class DailySafetyLogInput(CamelModel):
    site_id: int = Field(gt=0)
    date: str
    tbm: dict[str, Any] = Field(default_factory=dict)
    deadline: dict[str, Any] = Field(default_factory=dict)
    daily_safety_log: dict[str, Any] = Field(default_factory=dict)


class DailySafetyLogResult(CamelModel):
    site_id: int
    date: str
    summary: str
    tbm: str
    deadline: str
    daily_safety_log: str
    warnings: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
