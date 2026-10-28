"""도구 레지스트리: ToolInvoker 규약(tool_contract)의 구현.

A·C·D는 자기 도구를 `registry.register(...)`로 등록만 하면 에이전트가 이름으로 부른다.
mock을 실제 모듈로 바꿀 때도 같은 이름으로 다시 등록하면 된다 (replace=True).

    registry = ToolRegistry()
    registry.register("search_accidents", search_accidents, owner="A", description="...")
    registry.call("search_accidents", {"query": "...", "k": 5})

도구 함수 규약: `handler(arguments: Mapping[str, Any]) -> dict | BaseModel`
  - 예상된 실패는 ToolError(code, 고정 문구)로 던진다.
  - 프로그래밍 오류는 그대로 올린다 (레지스트리도 잡지 않는다).

RecordingToolInvoker는 호출마다 agent_steps용 기록을 남긴다. 기록은 on_call 콜백으로 넘기고,
콜백이 실패해도 도구 호출 결과는 그대로 돌려준다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel
from .tool_contract import ToolInvoker
from app.llm.llm_types import ToolSpec
from app.exceptions import ToolError
logger = logging.getLogger(__name__)

ToolHandler = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    handler: ToolHandler
    owner: str  # A, B, C, D
    description: str
    parameters: Mapping[str, Any]
    mock: bool

    def spec(self) -> ToolSpec:
        """LLM tool use에 넘길 설명 (2주차 LLM 루프에서 쓴다)."""
        return ToolSpec(name=self.name, description=self.description, parameters=dict(self.parameters))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        owner: str,
        description: str,
        parameters: Mapping[str, Any] | None = None,
        mock: bool = False,
        replace: bool = False,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("도구 이름이 비어 있습니다.")
        if not callable(handler):
            raise ValueError(f"도구 {name!r}의 handler가 호출 가능하지 않습니다.")
        if name in self._tools and not replace:
            raise ValueError(f"도구 {name!r}가 이미 등록되어 있습니다. 바꾸려면 replace=True.")
        self._tools[name] = RegisteredTool(
            name=name,
            handler=handler,
            owner=owner,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            mock=mock,
        )

    def has(self, name: str) -> bool:
        return name in self._tools

    def tools(self) -> list[RegisteredTool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools()]

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(
                code="NOT_FOUND", 
                message=f"'{name}' 도구가 등록되어 있지 않습니다.", 
                tool=name,
                status_code=404)
        return tool.handler(arguments)


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    input: dict[str, Any]
    output: dict[str, Any]
    elapsed_ms: int
    ok: bool
    error_code: str | None = None


def to_plain(value: Any) -> Any:
    """기록·응답용으로 Pydantic 모델을 dict로 바꾼다."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


class RecordingToolInvoker:
    """ToolInvoker를 감싸 호출마다 ToolCallRecord를 on_call로 넘긴다. 반환값·예외는 그대로 통과시킨다."""

    def __init__(self, inner: ToolInvoker, on_call: Callable[[ToolCallRecord], None]) -> None:
        self._inner = inner
        self._on_call = on_call

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            result = self._inner.call(name, arguments)
        except ToolError as exc:
            self._emit(
                ToolCallRecord(
                    name,
                    dict(arguments),
                    {"code": exc.code, "message": exc.message},
                    _ms(started),
                    ok=False,
                    error_code=exc.code,
                )
            )
            raise
        plain = to_plain(result)
        self._emit(ToolCallRecord(name, dict(arguments), plain if isinstance(plain, dict) else {"result": plain}, _ms(started), ok=True))
        return result

    def _emit(self, record: ToolCallRecord) -> None:
        try:
            self._on_call(record)
        except Exception:  # 기록 실패가 도구 호출을 막지 않게 한다
            logger.warning("도구 호출 기록 실패: tool=%s", record.tool_name, exc_info=True)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
