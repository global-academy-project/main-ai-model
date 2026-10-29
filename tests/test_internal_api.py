"""Spring 내부 도구 API 클라이언트: 요청 모양, 오류 매핑, 재시도."""

import pytest

from app.exceptions import ToolError
from app.integrations import InternalApiClient
from app.integrations.internal_api import HttpResult

from .fakes import FakeSpring, error, ok, timeout

TOKEN = "t" * 32


def client(fake: FakeSpring) -> InternalApiClient:
    return InternalApiClient("http://spring:8080/", TOKEN, connect_timeout=2.0, read_timeout=3.0, transport=fake)


def test_request_shape_and_data():
    fake = FakeSpring(ok({"result": "PASS"}))
    assert client(fake).call_tool("check_rules", {"siteId": 1}, retry=False) == {"result": "PASS"}
    req = fake.requests[0]
    assert req["url"] == "http://spring:8080/internal/v1/tools/check_rules"   # 끝 / 정리
    assert req["headers"]["X-Internal-Token"] == TOKEN
    assert req["body"] == {"siteId": 1}
    assert (req["connect_timeout"], req["read_timeout"]) == (2.0, 3.0)


@pytest.mark.parametrize("status,code,expected", [
    (400, "INVALID_REQUEST", "TOOL_INPUT_INVALID"),
    (404, "SITE_NOT_FOUND", "TOOL_INPUT_INVALID"),       # 같은 404라도 현장 문제는 입력 문제
    (404, "NOT_FOUND", "TOOL_TARGET_NOT_FOUND"),
    (401, "UNAUTHORIZED", "TOOL_UNAVAILABLE"),           # 토큰 설정 문제
    (500, "INTERNAL_ERROR", "TOOL_UNAVAILABLE"),
])
def test_error_mapping_by_code(status, code, expected):
    with pytest.raises(ToolError) as exc:
        client(FakeSpring(error(status, code))).call_tool("check_rules", {}, retry=False)
    assert exc.value.code == expected
    assert "원문" not in exc.value.message          # Spring 원문은 로그에만


def test_malformed_response_is_unavailable():
    with pytest.raises(ToolError) as exc:
        client(FakeSpring(HttpResult(502, b"<html>bad gateway</html>"))).call_tool("check_rules", {}, retry=True)
    assert exc.value.code == "TOOL_UNAVAILABLE"


def test_retry_once_on_timeout_when_allowed():
    fake = FakeSpring(timeout(), ok({"result": "PASS"}))
    assert client(fake).call_tool("check_rules", {}, retry=True) == {"result": "PASS"}
    assert len(fake.requests) == 2


def test_no_retry_when_saving():
    fake = FakeSpring(timeout(), ok({"result": "PASS"}))
    with pytest.raises(ToolError) as exc:
        client(fake).call_tool("check_rules", {}, retry=False)
    assert exc.value.code == "TOOL_UNAVAILABLE"
    assert len(fake.requests) == 1


def test_http_error_is_not_retried():
    fake = FakeSpring(error(500, "INTERNAL_ERROR"), ok({}))
    with pytest.raises(ToolError):
        client(fake).call_tool("check_rules", {}, retry=True)
    assert len(fake.requests) == 1


def test_config_required():
    with pytest.raises(ValueError):
        InternalApiClient("spring:8080", TOKEN, connect_timeout=2, read_timeout=3)
    with pytest.raises(ValueError):
        InternalApiClient("http://spring:8080", "", connect_timeout=2, read_timeout=3)
