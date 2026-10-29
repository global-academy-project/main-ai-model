"""Spring 내부 도구 API 공통 클라이언트 (계획서 7-2장 도구 호출 규격).

    POST {INTERNAL_API_BASE_URL}/internal/v1/tools/{도구이름}
    헤더 X-Internal-Token: {INTERNAL_API_TOKEN}
    응답 {"success": true, "data": {...}} / {"success": false, "error": {"code", "message", "details"}}

AI 서버는 업무 DB에 직접 붙지 않는다. 업무 기록이 필요하면 이 클라이언트로 Spring 도구를 부른다 (7-0장).
A·D가 기록 조회 도구를 붙일 때도 이 클라이언트를 쓴다.

오류 매핑 (HTTP 상태가 아니라 Spring error.code 기준. 같은 404 안에 두 경우가 있다)
    INVALID_REQUEST, SITE_NOT_FOUND     -> TOOL_INPUT_INVALID     (입력·현장 문제)
    NOT_FOUND                           -> TOOL_TARGET_NOT_FOUND  (판정을 저장할 문서가 없음)
    그 밖 (401 토큰 설정, 5xx, 연결 실패, 시간 초과, 응답 모양 이상) -> TOOL_UNAVAILABLE
ToolError message는 고정 문구만 쓴다. Spring 원문·토큰·응답 본문은 로그에만 남긴다 (tool_errors 규약).

재시도: retry=True일 때 연결 실패·시간 초과만 한 번 더. 저장하는 호출(retry=False)은 재시도하지 않는다 (판정 중복 방지).
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from app.exceptions import ToolError

logger = logging.getLogger(__name__)

TOOL_PATH = "/internal/v1/tools/"

_INPUT_CODES = frozenset({"INVALID_REQUEST", "SITE_NOT_FOUND"})
_TARGET_CODES = frozenset({"NOT_FOUND"})


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


class TransportError(Exception):
    """연결 실패·시간 초과 (재시도 대상)."""


# (url, 본문, 헤더, 연결 제한 초, 읽기 제한 초) -> HttpResult. 테스트는 가짜를 넣는다.
Transport = Callable[[str, bytes, Mapping[str, str], float, float], HttpResult]


def stdlib_transport(url: str, body: bytes, headers: Mapping[str, str],
                     connect_timeout: float, read_timeout: float) -> HttpResult:
    """표준 라이브러리 http.client. 연결과 읽기 시간 제한을 따로 건다."""
    parts = urlsplit(url)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=connect_timeout)
    try:
        conn.connect()
        conn.sock.settimeout(read_timeout)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        conn.request("POST", path, body=body, headers=dict(headers))
        res = conn.getresponse()
        return HttpResult(res.status, res.read())
    except (OSError, socket.timeout, http.client.HTTPException) as exc:
        raise TransportError(type(exc).__name__) from exc
    finally:
        conn.close()


class InternalApiClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float,
        read_timeout: float,
        transport: Transport = stdlib_transport,
    ) -> None:
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise ValueError("INTERNAL_API_BASE_URL은 http:// 또는 https:// 로 시작해야 합니다.")
        if not token:
            raise ValueError("INTERNAL_API_TOKEN이 필요합니다.")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._transport = transport

    def call_tool(self, tool: str, payload: Mapping[str, Any], *, retry: bool) -> dict[str, Any]:
        """Spring 도구를 부르고 data를 돌려준다. 실패는 ToolError."""
        url = f"{self._base_url}{TOOL_PATH}{tool}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Internal-Token": self._token,
        }
        attempts = 2 if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                result = self._transport(url, body, headers, self._connect_timeout, self._read_timeout)
                break
            except TransportError as exc:
                logger.warning("내부 API 연결 실패: tool=%s attempt=%d/%d reason=%s", tool, attempt, attempts, exc)
                if attempt == attempts:
                    raise _unavailable(tool) from exc
        return _read(tool, result)


def _read(tool: str, result: HttpResult) -> dict[str, Any]:
    try:
        envelope = json.loads(result.body.decode("utf-8")) if result.body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        envelope = None
    if not isinstance(envelope, dict):
        logger.warning("내부 API 응답 모양이 다름: tool=%s status=%d", tool, result.status)
        raise _unavailable(tool)

    if 200 <= result.status < 300 and envelope.get("success") is True and isinstance(envelope.get("data"), dict):
        return envelope["data"]

    error = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
    code = error.get("code")
    logger.warning("내부 API 오류: tool=%s status=%d code=%s message=%s",
                   tool, result.status, code, error.get("message"))
    if code in _INPUT_CODES:
        raise ToolError(code="TOOL_INPUT_INVALID", message=f"{tool} 입력을 서버가 받지 않았습니다.",
                        status_code=400, tool=tool)
    if code in _TARGET_CODES:
        raise ToolError(code="TOOL_TARGET_NOT_FOUND", message=f"{tool} 대상 문서를 찾을 수 없습니다.",
                        status_code=404, tool=tool)
    raise _unavailable(tool)


def _unavailable(tool: str) -> ToolError:
    return ToolError(code="TOOL_UNAVAILABLE", message=f"{tool} 서버를 지금 사용할 수 없습니다.",
                     status_code=503, tool=tool)
