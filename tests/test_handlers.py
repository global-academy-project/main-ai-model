import asyncio
import json

import pytest
from fastapi.exceptions import RequestValidationError

from app.exceptions.app_errors import AppError
from app.exceptions.business_errors import (
    BusinessError, EngineNotAllowedError, FileTooLargeError, PageLimitExceededError, UnsupportedFileTypeError,
)
from app.handlers.exception_handlers import (
    app_error_handler, business_error_handler, unexpected_error_handler, validation_error_handler,
)


def call(handler, exc):
    resp = asyncio.run(handler(None, exc))
    return resp.status_code, json.loads(resp.body)


@pytest.mark.parametrize("exc,status,code,reason_part", [
    (FileTooLargeError(20), 413, "FILE_TOO_LARGE", "20MB"),
    (UnsupportedFileTypeError("PDF"), 400, "UNSUPPORTED_FILE_TYPE", "PDF"),
    (PageLimitExceededError(50), 400, "PAGE_LIMIT_EXCEEDED", "50"),
    (EngineNotAllowedError("OPENAI"), 400, "ENGINE_NOT_ALLOWED", "OPENAI"),
])
def test_business_errors_use_error_envelope(exc, status, code, reason_part):
    got_status, body = call(business_error_handler, exc)
    assert got_status == status
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert reason_part in body["error"]["details"][0]["reason"]
    assert body["error"]["details"][0]["field"] is None


def test_business_error_without_details_returns_empty_list_not_null():
    exc = BusinessError(code="X", message="m", status_code=409)
    status, body = call(business_error_handler, exc)
    assert status == 409
    assert body["error"]["details"] == []


def test_app_error_uses_error_envelope():
    status, body = call(app_error_handler, AppError(code="PDF_UNREADABLE", message="열 수 없음", status_code=422))
    assert status == 422
    assert body == {"success": False, "error": {"code": "PDF_UNREADABLE", "message": "열 수 없음", "details": []}}


def test_app_error_requires_keyword_arguments():
    # 위치 인자로 호출하면 TypeError. (코드 곳곳의 AppError 호출 방식과 일치하는지 확인하는 기준)
    with pytest.raises(TypeError):
        AppError(503, "CODE", "message")


def test_unexpected_error_hides_internal_message():
    status, body = call(unexpected_error_handler, RuntimeError("db password=abc host=internal"))
    assert status == 500
    assert body["code"] == "INTERNAL_SERVER_ERROR"
    assert "password" not in json.dumps(body)


def test_validation_error_returns_422_with_code():
    exc = RequestValidationError([{"type": "missing", "loc": ("body", "file"), "msg": "Field required", "input": None}])
    status, body = call(validation_error_handler, exc)
    assert status == 422
    assert body["code"] == "VALIDATION_ERROR"
    assert body["details"][0]["loc"] == ["body", "file"]


@pytest.mark.xfail(strict=False, reason="검증 오류(422)와 예기치 못한 오류(500)는 {success:false, error:{...}} 봉투가 아니라 평평한 JSON 으로 내려간다")
@pytest.mark.parametrize("handler,exc", [
    (validation_error_handler, RequestValidationError([{"type": "missing", "loc": ("body",), "msg": "x", "input": None}])),
    (unexpected_error_handler, RuntimeError("x")),
])
def test_all_errors_share_the_same_envelope(handler, exc):
    _, body = call(handler, exc)
    assert body.get("success") is False and "error" in body
