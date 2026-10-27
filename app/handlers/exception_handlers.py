import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from app.schemas import ErrorDetail, ErrorResponse, ErrorBody
from app.exceptions import BusinessError, LLMError, to_error_info

logger = logging.getLogger(__name__)

"""예외 핸들러.

예외를 에러 응답(`ErrorResponse`)으로 변환하는 FastAPI 핸들러 모음.
"""

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
    logger.info("ErrorResponse 발생: code=%s", code)
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
    logger.warning("LLM 오류: type=%s code=%s path=%s",
                   type(exc).__name__, info.code, request.url.path, exc_info=exc)
    return build_error_response(
        status=info.status,
        code=info.code,
        message=info.message,
        details=[ErrorDetail(reason=d) for d in info.details],   # 필드명은 확인 필요
    )