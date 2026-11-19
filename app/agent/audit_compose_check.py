"""구성기 입력(stats) 검증과 문장 숫자 대조기.

- 값이 이상하면(정수가 아님, 음수, 서로 모순) 그 항목은 '확인 불가'로 둔다. 0으로 바꾸지 않는다.
- 비율은 건수에서 코드가 다시 계산한다(반올림 HALF_UP, 소수 1자리). Spring이 보낸 비율과 다르면 경고만 남긴다.
- 문장 대조기는 날짜(YYYY-MM-DD)를 숫자로 읽지 않는다.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"(?<![0-9.])\d+(?:\.\d+)?")

KEY_LABELS: dict[str, str] = {
    "works": "작업",
    "written": "평가 작성",
    "approved": "승인",
    "missing": "평가 없음",
    "permits": "허가",
    "requested": "승인 요청",
    "rejected": "반려",
    "canceled": "취소",
    "blocked": "차단",
    "actions": "개선조치",
    "completed": "완료",
    "verified": "검증 완료",
    "open": "미완료",
    "overdue": "기한 경과",
    "count": "건수",
}


class StatsInvalid(Exception):
    """집계 값을 믿을 수 없을 때. `code`는 STATS_INVALID 또는 STATS_INCONSISTENT."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def percent(part: int, whole: int) -> float | None:
    """part ÷ whole × 100, 소수 1자리 HALF_UP. whole이 0이면 None."""
    if whole == 0:
        return None
    value = (Decimal(part) * 100 / Decimal(whole)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(value)


def read_ints(stats: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StatsInvalid("STATS_INVALID", f"{KEY_LABELS.get(key, key)} 값이 올바른 건수가 아님")
        out[key] = value
    return out


def _inconsistent(message: str) -> StatsInvalid:
    return StatsInvalid("STATS_INCONSISTENT", message)


def check_risk(stats: dict[str, Any]) -> dict[str, int]:
    v = read_ints(stats, ("works", "written", "approved", "missing"))
    if v["written"] > v["works"] or v["approved"] > v["written"] or v["missing"] != v["works"] - v["written"]:
        raise _inconsistent("위험성평가 건수(작업·평가 작성·승인·평가 없음)가 서로 맞지 않음")
    return v


def check_permit(stats: dict[str, Any]) -> dict[str, int]:
    v = read_ints(stats, ("permits", "requested", "approved", "rejected", "canceled", "blocked"))
    if v["approved"] + v["rejected"] + v["canceled"] > v["requested"]:
        raise _inconsistent("작업허가 건수(승인·반려·취소가 승인 요청보다 많음)가 서로 맞지 않음")
    return v


def check_count(stats: dict[str, Any]) -> dict[str, int]:
    return read_ints(stats, ("count",))


def check_corrective(stats: dict[str, Any]) -> dict[str, int]:
    v = read_ints(stats, ("actions", "completed", "verified", "open", "overdue"))
    if (
        v["completed"] > v["actions"]
        or v["verified"] > v["completed"]
        or v["open"] != v["actions"] - v["completed"]
        or v["overdue"] > v["open"]
    ):
        raise _inconsistent("개선조치 건수(등록·완료·검증·미완료·기한 경과)가 서로 맞지 않음")
    return v


def rate_differs(provided: Any, computed: float | None) -> bool:
    """Spring이 보낸 비율이 코드가 계산한 값과 다른지. 보낸 값이 없으면(키 없음) 비교하지 않는다."""
    if provided is None and computed is None:
        return False
    if isinstance(provided, bool) or not isinstance(provided, (int, float)):
        return provided is not None or computed is not None
    if computed is None:
        return True
    return abs(float(provided) - computed) > 0.05


def fmt(value: float | int) -> str:
    """문장·허용 집합에서 같은 방식으로 쓰는 숫자 표기. 정수는 그대로, 비율은 소수 1자리."""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def numbers_in(text: str) -> list[str]:
    """문장에서 숫자를 뽑는다. 날짜(YYYY-MM-DD)는 먼저 지운다."""
    return _NUMBER.findall(_DATE.sub(" ", text))


def unexpected_numbers(text: str, allowed: set[str]) -> list[str]:
    return [n for n in numbers_in(text) if n not in allowed]
