"""Spring 내부 도구 API를 호출하는 작은 HTTP 클라이언트."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.exceptions import ToolError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


class TransportError(Exception):
    """HTTP 연결 자체가 완료되지 못한 경우."""


Transport = Callable[[str, bytes, Mapping[str, str], float, float], HttpResult]


class InternalApiClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = 2.0,
        read_timeout: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("internal_api_base_url은 http 또는 https URL이어야 합니다.")
        if not token.strip():
            raise ValueError("internal_api_token이 비어 있습니다.")
        if not token.isascii():
            raise ValueError("internal_api_token은 ASCII 문자열이어야 합니다.")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._transport = transport or self._request

    def call_tool(self, tool_name: str, body: Mapping[str, object], *, retry: bool = True) -> dict:
        url = f"{self._base_url}/internal/v1/tools/{tool_name}"
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Internal-Token": self._token,
        }
        attempts = 2 if retry else 1
        for attempt in range(attempts):
            try:
                result = self._transport(
                    url, encoded, headers, self._connect_timeout, self._read_timeout
                )
                return self._read_response(tool_name, result)
            except TransportError as exc:
                if attempt + 1 == attempts:
                    logger.warning("Spring 내부 API 연결 실패: tool=%s", tool_name, exc_info=exc)
                    raise ToolError(
                        "TOOL_UNAVAILABLE",
                        "Spring 기록 조회 도구에 연결할 수 없습니다.",
                        tool=tool_name,
                    ) from exc
            except ToolError:
                raise
            except Exception as exc:
                logger.warning("Spring 내부 API 호출 실패: tool=%s", tool_name, exc_info=exc)
                raise ToolError(
                    "TOOL_UNAVAILABLE",
                    "Spring 기록 조회 도구 호출에 실패했습니다.",
                    tool=tool_name,
                ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _read_response(tool_name: str, result: HttpResult) -> dict:
        try:
            payload = json.loads(result.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolError(
                "TOOL_UNAVAILABLE",
                "Spring 기록 조회 응답을 읽을 수 없습니다.",
                tool=tool_name,
            ) from exc
        if not isinstance(payload, Mapping):
            raise ToolError(
                "TOOL_UNAVAILABLE", "Spring 기록 조회 응답 형식이 올바르지 않습니다.", tool=tool_name
            )
        if not 200 <= result.status < 300 or payload.get("success") is not True:
            error = payload.get("error")
            code = error.get("code") if isinstance(error, Mapping) else None
            if code == "NOT_FOUND":
                internal_code = "TOOL_TARGET_NOT_FOUND"
            elif code in {"INVALID_REQUEST", "SITE_NOT_FOUND"}:
                internal_code = "TOOL_INPUT_INVALID"
            else:
                internal_code = "TOOL_UNAVAILABLE"
            raise ToolError(
                internal_code, "Spring 기록 조회를 완료하지 못했습니다.", tool=tool_name
            )
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ToolError(
                "TOOL_OUTPUT_INVALID",
                "Spring 기록 조회 결과 형식이 올바르지 않습니다.",
                tool=tool_name,
            )
        return dict(data)

    @staticmethod
    def _request(
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        connect_timeout: float,
        read_timeout: float,
    ) -> HttpResult:
        request = Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urlopen(request, timeout=max(connect_timeout, read_timeout)) as response:
                return HttpResult(response.status, response.read())
        except HTTPError as exc:
            return HttpResult(exc.code, exc.read())
        except (URLError, TimeoutError, OSError) as exc:
            raise TransportError(str(exc)) from exc
