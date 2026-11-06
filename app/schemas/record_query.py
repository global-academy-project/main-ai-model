"""Spring과 AI 서버 사이의 기록질문 계약."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .base_schemas import CamelModel


class RecordQueryRequest(CamelModel):
    run_id: int = Field(strict=True, gt=0)
    site_id: int = Field(strict=True, gt=0)
    message: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any] | None = None
    mode: Literal["VERIFY", "GUIDE"]


class RecordQueryAsk(CamelModel):
    message: str = Field(min_length=1)


class RecordQueryItem(CamelModel):
    query_type: str = Field(min_length=1)
    conditions: dict[str, Any]
    total: int = Field(ge=0)
    truncated: bool = False


class RecordEvidence(CamelModel):
    type: Literal[
        "WORK", "PERMIT", "RISK", "TBM", "INSPECTION", "APPROVAL",
        "CORRECTIVE_ACTION", "NEAR_MISS", "DAILY_LOG"
    ]
    id: int = Field(strict=True, gt=0)
    label: str = Field(min_length=1)


class RecordQueryStep(CamelModel):
    seq: int = Field(ge=1)
    kind: Literal["TOOL", "PHASE"]
    name: str
    tool_name: str | None = None
    status: Literal["OK", "ERROR"]
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = Field(default=0, ge=0)
    error_code: str | None = None


class RecordQueryResult(CamelModel):
    run_status: Literal["WAITING_INPUT", "DONE"]
    answer: str | None = None
    ask: RecordQueryAsk | None = None
    queries: list[RecordQueryItem] = Field(default_factory=list)
    evidence: list[RecordEvidence] = Field(default_factory=list)
    context: dict[str, Any] | None = None
    steps: list[RecordQueryStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_payload(self) -> "RecordQueryResult":
        if self.run_status == "DONE" and not (self.answer and self.answer.strip()):
            raise ValueError("DONE 응답에는 answer가 필요합니다.")
        if self.run_status == "WAITING_INPUT" and self.ask is None:
            raise ValueError("WAITING_INPUT 응답에는 ask가 필요합니다.")
        return self
