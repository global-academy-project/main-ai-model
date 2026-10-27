"""엔진에 독립적인 공통 메시지·도구 타입. 에이전트 루프는 이 타입만 안다."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class ToolSpec(BaseModel):
    """LLM에게 알려주는 도구 설명. parameters는 JSON Schema."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class Message(BaseModel):
    """role=tool 이면 tool_call_id와 content(도구 결과 문자열)를 채운다."""
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant 전용
    tool_call_id: str | None = None                           # tool 전용

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Literal["end", "tool_use", "max_tokens"] = "end"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
