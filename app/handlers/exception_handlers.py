import logging
from fastapi import Request
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import JSONResponse
from app.schemas.error import ErrorDetail, ErrorResponse, ErrorBody
from app.exceptions.business_errors import BusinessError
from app.exceptions.app_errors import AppError

"""예외 핸들러.

예외를 에러 응답(`ErrorResponse`)으로 변환하는 FastAPI 핸들러 모음.
"""
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

async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    
    return JSONResponse(
        status_code=422,
        content={
            "code": "VALIDATION_ERROR",
            "message": (
                "요청 데이터가 올바르지 않습니다."
            ),
            "status": 422,
            "details": exc.errors(),
        },
    )

async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": (
                "서버 내부 오류가 발생했습니다."
            ),
            "status": 500,
            "details": None,
        },
    )

async def app_error_handler(
    request: Request,
    exc: AppError
) -> JSONResponse:
    
    return build_error_response(
        status=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )