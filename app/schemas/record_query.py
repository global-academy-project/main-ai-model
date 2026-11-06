"""Spring ↔ AI 서버 기록질문 계약."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base_schemas import CamelModel


class RecordQueryRequest(CamelModel):
    run_id: int = Field(gt=0)
    site_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any] | None = None
    mode: str = "VERIFY"


class RecordQueryResult(CamelModel):
    run_status: str
    answer: str | None = None
    ask: dict[str, Any] | None = None
    queries: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
