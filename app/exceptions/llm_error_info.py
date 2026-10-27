"""LLM 예외를 응답용 정보로 바꾸는 매핑.

FastAPI를 알지 못한다. HTTP 핸들러와 에이전트 루프 러너가 같은 표를 쓰도록 분리했다.
`message`는 사용자에게 내려가는 고정 문구이며, 예외 원문(`str(exc)`)은 넣지 않는다.
"""
from collections.abc import Callable
from dataclasses import dataclass

from .llm_errors import EngineBlockedError, LLMConfigError, LLMError


@dataclass(frozen=True)
class LLMErrorInfo:
    status: int    # HTTP 경로에서만 의미가 있다. 루프 러너는 무시한다.
    code: str
    message: str
    details: tuple[str, ...] = ()

_FALLBACK = LLMErrorInfo(502, "LLM_CALL_FAILED", "AI 엔진 호출에 실패했습니다.")

def _engine_blocked(exc: LLMError) -> LLMErrorInfo:
    return LLMErrorInfo(
        503,
        "LLM_ENGINE_BLOCKED",
        "엔진을 사용할 수 없습니다",
        details=f"현재 환경에서는 '{exc.engine}' 엔진을 사용할 수 없습니다.",
    )

def _engine_config_error(exc: LLMError) -> LLMErrorInfo:
    assert isinstance(exc, LLMConfigError)
    return LLMErrorInfo(
        500,
        "LLM_CONFIG_ERROR",
        "AI 엔진 설정에 문제가 있습니다.",
        details=(str(exc),),
    )


# 자식 클래스를 부모보다 먼저 둔다 (isinstance는 첫 일치에서 멈춘다).
_RULES: list[tuple[type[LLMError], Callable[[LLMError], LLMErrorInfo]]] = [
    (EngineBlockedError, _engine_blocked),
    (LLMConfigError, _engine_config_error),
]


def to_error_info(exc: LLMError) -> LLMErrorInfo:
    """예외 종류에 맞는 응답 정보를 돌려준다. 어디에도 해당하지 않으면 502."""
    for exc_type, build in _RULES:
        if isinstance(exc, exc_type):
            return build(exc)
    return _FALLBACK