from typing import Literal
from pydantic import BaseModel, Field

SummaryLength = Literal[
    "short",
    "medium",
    "long",
]

SummaryTone = Literal[
    "neutral",
    "professional",
    "friendly",
]

class SummaryCreateRequest(BaseModel):
    text: str = Field(
        min_length=10,
        max_length=20_000,
    )
    length: SummaryLength = "medium"
    tone: SummaryTone = "neutral"

class SummaryResponse(BaseModel):
    summary: str
    length: SummaryLength
    tone: SummaryTone
    status: Literal["completed"]