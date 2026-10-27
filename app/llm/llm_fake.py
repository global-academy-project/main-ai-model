"""테스트·목업용 가짜 클라이언트. 미리 짠 응답을 순서대로 돌려준다."""
from __future__ import annotations
from typing import Callable, Union
from .llm_base import LLMClient
from llm_types import LLMResponse, Message, ToolCall, ToolSpec

Step = Union[LLMResponse, Callable[[list[Message]], LLMResponse]]

def text_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, stop_reason="end", model="fake")

def tool_response(name: str, arguments: dict | None = None, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments or {})],
        stop_reason="tool_use",
        model="fake",
    )

class FakeLLMClient(LLMClient):
    name = "fake"

    def __init__(self, script: list[Step] | None = None):
        self.script = list(script or [])
        self.calls: list[dict] = []   # 테스트에서 무엇을 받았는지 확인용

    def chat(self, messages, *, system="", tools=None, max_tokens=None) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system,
                           "tools": tools, "max_tokens": max_tokens})
        if not self.script:
            return text_response("[fake] 스크립트가 비어 있어 기본 응답을 반환합니다.")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step
