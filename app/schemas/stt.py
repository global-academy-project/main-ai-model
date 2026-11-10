"""음성 → 글 응답 (Spring SttResult 와 같은 모양: text, language, model)."""

from __future__ import annotations

from pydantic import Field

from .base_schemas import CamelModel


class SttTranscript(CamelModel):
    text: str = Field(description="받아 적은 글 (표기 맞춤 후). 알아들은 말이 없으면 빈 글자")
    language: str = Field(description="언어 코드 (예: ko)")
    model: str = Field(description="받아 적은 모델 이름")
