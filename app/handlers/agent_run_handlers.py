"""에이전트 실행 경로의 오류 응답: 실행 실패(+steps), 요청 검증 실패(422), 처리되지 않은 오류(500).

모든 응답은 공통 모양 `{"success": false, "error": {"code", "message", "details": []}}`이다.
실행 실패만 최상위에 `steps`를 덧붙인다 (Spring이 실패한 실행의 단계도 agent_steps로 저장한다. 계획서 7-2).

- 번역 규칙은 기존 것을 그대로 쓴다. LLMError는 기존 `llm_error_handler`, ToolError는 `to_external_error`.
- 내부 코드(TOOL_UNAVAILABLE 등)와 원문 오류는 로그에만 남긴다.
"""

from __future__ import annotations

import inspect
import json
import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import AgentRunError, LLMError
from app.exceptions.error_catalog import ExternalCode, to_external_error
from app.schemas.error_schemas import ErrorBody, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def _error_response(status: int, code: str, message: str, details: list[ErrorDetail] | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details or []))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


async def agent_run_error_handler(request: Request, exc: AgentRunError) -> JSONResponse:
    cause = exc.cause
    if isinstance(cause, LLMError):
        # 기존 LLM 오류 응답을 그대로 쓴다 (번역 규칙을 두 곳에 두지 않는다)
        from app.handlers import llm_error_handler  # 패키지 __init__과의 순환 임포트를 피한다

        base = llm_error_handler(request, cause)
        if inspect.isawaitable(base):
            base = await base
        status, body = base.status_code, json.loads(base.body)
    else:
        mapped = to_external_error(cause.code, cause.status_code)
        logger.warning(
            "에이전트 실행 실패: path=%s code=%s tool=%s",
            request.url.path,
            cause.code,
            cause.tool,
            exc_info=cause,
        )
        status = mapped.status
        body = ErrorResponse(error=ErrorBody(code=mapped.code, message=cause.message)).model_dump(mode="json")

    body["steps"] = [step.model_dump(mode="json", by_alias=True) for step in exc.steps]
    return JSONResponse(status_code=status, content=body)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI 기본 `{"detail": ...}` 대신 공통 모양. 입력 값(input)은 싣지 않는다 (요청 원문이 새지 않게)."""
    details = [
        ErrorDetail(field=".".join(str(p) for p in err["loc"][1:]) or None, reason=err["msg"])
        for err in exc.errors()
    ]
    logger.info("요청 검증 실패: path=%s fields=%s", request.url.path, [d.field for d in details])
    return _error_response(422, ExternalCode.VALIDATION_ERROR, "요청 형식이 올바르지 않습니다.", details)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """처리되지 않은 예외. 원문은 로그에만 남기고 고정 문구로 답한다.

    Starlette는 이 응답을 보낸 뒤 예외를 다시 올려 서버 로그에 traceback이 한 번 더 찍힌다 (정상).
    """
    logger.error("처리되지 않은 오류: path=%s", request.url.path, exc_info=exc)
    return _error_response(500, ExternalCode.INTERNAL_ERROR, "AI 서버 내부 오류로 처리하지 못했습니다.")
