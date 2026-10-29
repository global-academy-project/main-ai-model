"""오류 코드 목록과 번역 표.

AI 서버 안에서 쓰는 내부 코드(`InternalCode`)와 Spring에 내보내는 바깥 코드(`ExternalCode`)를 한 곳에 모은다.
FastAPI를 알지 못한다 (exceptions 패키지 규칙). 핸들러가 `to_external_error`로 번역해서 응답을 만든다.

  - 내부 코드: 로그, agent_steps, 테스트에서 원인을 구분하는 용도. 응답에는 나가지 않는다.
  - 바깥 코드: Spring과 미리 정한 코드만 쓴다. 아래 값은 가칭이며, Spring 목록을 받으면 `ExternalCode`만 고친다.

새 내부 코드를 만들면 `InternalCode`와 `EXTERNAL_BY_INTERNAL`을 같이 고친다 (tests/test_error_catalog.py가 빠뜨림을 잡는다).
기준: 500 = 우리 쪽 코드·설정 문제, 502 = 도구·내부 API·모델 쪽 문제, 503 = 현재 환경에서 막힘.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping, NamedTuple


class InternalCode:
    """AI 서버 내부 오류 코드 (ToolError / LLMError 계열)."""

    # 도구
    TOOL_NOT_FOUND: Final = "TOOL_NOT_FOUND"
    TOOL_INPUT_INVALID: Final = "TOOL_INPUT_INVALID"
    TOOL_TARGET_NOT_FOUND: Final = "TOOL_TARGET_NOT_FOUND"
    TOOL_UNAVAILABLE: Final = "TOOL_UNAVAILABLE"
    TOOL_OUTPUT_INVALID: Final = "TOOL_OUTPUT_INVALID"
    # 서식
    FORM_INVALID: Final = "FORM_INVALID"
    FORM_TYPE_UNKNOWN: Final = "FORM_TYPE_UNKNOWN"
    # 근거 검색
    EVIDENCE_SEARCH_FAILED: Final = "EVIDENCE_SEARCH_FAILED"
    # LLM (llm_error_info.py의 code와 같아야 한다)
    LLM_CONFIG_ERROR: Final = "LLM_CONFIG_ERROR"
    LLM_CALL_FAILED: Final = "LLM_CALL_FAILED"
    LLM_ENGINE_BLOCKED: Final = "LLM_ENGINE_BLOCKED"


class ExternalCode:
    """Spring에 내보내는 오류 코드. 가칭 — Spring 목록을 받으면 이 값만 바꾼다."""

    VALIDATION_ERROR: Final = "VALIDATION_ERROR"          # 요청 본문 검증 실패 (422)
    INTERNAL_ERROR: Final = "INTERNAL_ERROR"              # 우리 쪽 코드·설정 문제 (500)
    BAD_GATEWAY: Final = "BAD_GATEWAY"                    # 도구·내부 API·모델 문제 (502)
    SERVICE_UNAVAILABLE: Final = "SERVICE_UNAVAILABLE"    # 현재 환경에서 막힘 (503)


class ErrorMapping(NamedTuple):
    status: int
    code: str


EXTERNAL_BY_INTERNAL: Final[Mapping[str, ErrorMapping]] = MappingProxyType({
    # 500: 우리 쪽 코드·설정
    InternalCode.TOOL_NOT_FOUND: ErrorMapping(500, ExternalCode.INTERNAL_ERROR),
    InternalCode.TOOL_INPUT_INVALID: ErrorMapping(500, ExternalCode.INTERNAL_ERROR),
    InternalCode.FORM_INVALID: ErrorMapping(500, ExternalCode.INTERNAL_ERROR),
    InternalCode.FORM_TYPE_UNKNOWN: ErrorMapping(500, ExternalCode.INTERNAL_ERROR),
    InternalCode.LLM_CONFIG_ERROR: ErrorMapping(500, ExternalCode.INTERNAL_ERROR),
    # 502: 도구·내부 API·모델
    InternalCode.TOOL_UNAVAILABLE: ErrorMapping(502, ExternalCode.BAD_GATEWAY),
    InternalCode.TOOL_TARGET_NOT_FOUND: ErrorMapping(502, ExternalCode.BAD_GATEWAY),
    InternalCode.TOOL_OUTPUT_INVALID: ErrorMapping(502, ExternalCode.BAD_GATEWAY),
    InternalCode.EVIDENCE_SEARCH_FAILED: ErrorMapping(502, ExternalCode.BAD_GATEWAY),
    InternalCode.LLM_CALL_FAILED: ErrorMapping(502, ExternalCode.BAD_GATEWAY),
    # 503: 환경에서 막힘
    InternalCode.LLM_ENGINE_BLOCKED: ErrorMapping(503, ExternalCode.SERVICE_UNAVAILABLE),
})

# 표에 없는 코드는 외부 문제로 본다.
FALLBACK: Final = ErrorMapping(502, ExternalCode.BAD_GATEWAY)

# 상태를 직접 지정한 ToolError의 바깥 코드는 상태로 정한다 (상태와 코드가 어긋나지 않게).
EXTERNAL_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType({
    400: ExternalCode.VALIDATION_ERROR,
    422: ExternalCode.VALIDATION_ERROR,
    500: ExternalCode.INTERNAL_ERROR,
    502: ExternalCode.BAD_GATEWAY,
    503: ExternalCode.SERVICE_UNAVAILABLE,
})


def to_external_error(internal_code: str, status: int | None = None) -> ErrorMapping:
    """내부 코드(와 선택적으로 지정된 상태)를 (HTTP 상태, 바깥 코드)로 번역한다."""
    mapped = EXTERNAL_BY_INTERNAL.get(internal_code, FALLBACK)
    if status is None:
        return mapped
    return ErrorMapping(status, EXTERNAL_BY_STATUS.get(status, mapped.code))
