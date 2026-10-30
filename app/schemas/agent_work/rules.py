from __future__ import annotations
from typing import Literal
from pydantic import Field, BaseModel
from app.schemas import CamelModel

class RuleCheckDto(BaseModel):
    blocked: bool
    evaluations: list[RuleEvaluationDto]
    conflicts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

class RuleEvaluationDto(CamelModel):
    rule_code: str
    title: str
    status: Literal["PASS", "FAIL", "MISSING", "CHECK"]
    severity: Literal["BLOCK", "WARN"]
    message: str
    basis: str

