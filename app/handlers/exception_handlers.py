"""예외 핸들러.

예외를 에러 응답(`ErrorResponse`)으로 변환하는 FastAPI 핸들러 모음.

Spring에 나가는 `error.code`는 미리 정해진 코드만 쓴다. AI 서버 내부 코드(`TOOL_INPUT_INVALID` 등)는
`app/exceptions/error_catalog.py`의 표 한 곳에서 번역하고, 원래 코드는 로그에만 남긴다.
"""

import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from app.schemas import ErrorDetail, ErrorResponse, ErrorBody
from app.exceptions.error_catalog import to_external_error
from app.exceptions import (
    BusinessError,
    LLMError,
    ToolError,
    to_error_info
)

logger = logging.getLogger(__name__)

def build_error_response(
    *,
    status: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None
) -> JSONResponse:
    """`ErrorResponse` 형식의 JSON 응답을 만든다.

    Args:
        status: HTTP 상태 코드.
        code: 프론트 분기용 에러 식별자.
        message: 사용자에게 보여줄 수 있는 요약 메시지.
        details: 세부 사유 목록. None이면 빈 배열로 내려간다.

    Returns:
        `success: false`와 `error` 객체를 담은 JSONResponse.
    """
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details or []
        )
    )
    logger.info("ErrorResponse 발생: status=%s code=%s", status, code)
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))

async def business_error_handler(
    request: Request,
    exc: BusinessError
) -> JSONResponse:

    return build_error_response(
        status=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )

async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    info = to_error_info(exc)
    status, code = to_external_error(info.code, info.status)
    logger.warning("LLM 오류: type=%s internal_code=%s path=%s",
                   type(exc).__name__, info.code, request.url.path, exc_info=exc)
    return build_error_response(
        status=status,
        code=code,
        message=info.message,
        details=[ErrorDetail(reason=d) for d in info.details],
    )


async def tool_error_handler(request: Request, exc: ToolError) -> JSONResponse:
    status, code = to_external_error(exc.code, exc.status_code)
    logger.warning("도구 오류: internal_code=%s tool=%s path=%s",
                   exc.code, exc.tool, request.url.path)
    return build_error_response(
        status=status,
        code=code,
        message=exc.message,
    )
