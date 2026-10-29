"""테스트용 가짜 Spring 내부 API (네트워크 없이)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from app.integrations.internal_api import HttpResult, TransportError


class FakeSpring:
    """요청을 기록하고, 정해 둔 응답을 순서대로 돌려준다. 응답 대신 예외를 넣으면 그 예외를 낸다."""

    def __init__(self, *responses: HttpResult | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str],
                 connect_timeout: float, read_timeout: float) -> HttpResult:
        self.requests.append({"url": url, "body": json.loads(body), "headers": dict(headers),
                              "connect_timeout": connect_timeout, "read_timeout": read_timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def ok(data: dict[str, Any]) -> HttpResult:
    return HttpResult(200, json.dumps({"success": True, "data": data, "error": None}).encode())


def error(status: int, code: str) -> HttpResult:
    return HttpResult(status, json.dumps({"success": False, "data": None,
                                          "error": {"code": code, "message": "원문", "details": None}}).encode())


def timeout() -> TransportError:
    return TransportError("timeout")


def check(rule_code: str, target: str, status: str, *, actual: str | None = None, expected: str = "",
          severity: str = "BLOCK", message: str | None = None, category: str = "MEASUREMENT",
          legal_basis: str = "근거 조항") -> dict[str, Any]:
    """Spring RuleCheckResult.Check 한 줄."""
    return {"ruleCode": rule_code, "category": category, "target": target, "status": status, "actual": actual,
            "expected": expected, "severity": severity, "message": message,
            "action": None if status == "PASS" else "조치", "legalBasis": legal_basis, "sourceUrl": None}


def result(result_: str, checks: list[dict[str, Any]], *, evaluation_id: int | None = None) -> dict[str, Any]:
    """Spring RuleCheckResult (data)."""
    return {"evaluationId": evaluation_id, "result": result_, "rulesApplied": len(checks),
            "violations": [c for c in checks if c["status"] != "PASS"], "unusedFields": [], "checks": checks}
