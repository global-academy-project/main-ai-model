from __future__ import annotations
from abc import ABC, abstractmethod
from .llm_types import LLMResponse, Message, ToolSpec

class LLMClient(ABC):
    """모든 LLM 호출은 이 인터페이스 한 곳을 지난다."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...