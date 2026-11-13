"""사진 인식 응답 (Spring PhotoVision 과 같은 모양: model, people, helmetMissing, boxes)."""

from __future__ import annotations

from pydantic import Field

from .base_schemas import CamelModel


class VisionBox(CamelModel):
    label: str = Field(description="person | helmet | no_helmet")
    score: float | None = Field(default=None, description="0~1")
    x: float = Field(description="사진 너비에 대한 0~1 비율 (왼쪽)")
    y: float = Field(description="사진 높이에 대한 0~1 비율 (위)")
    w: float
    h: float


class TbmPhotoVision(CamelModel):
    model: str = Field(description="인식 모델 이름 (photo_checks.model 에 남는다)")
    people: int | None = Field(default=None, description="사진 속 인원. 판단 못 하면 null")
    helmet_missing: int | None = Field(default=None, description="안전모 미착용 인원(맨머리 수). 안전모 모델이 없으면 null")
    boxes: list[VisionBox] = Field(default_factory=list)
