"""발화에서 작업 날짜를 읽는다 (slots.workDate, YYYY-MM-DD).

Spring(AgentDraftWriter)은 slots.workDate로 작업·위험성평가·TBM 날짜를 정하고, 없으면 오늘 날짜로 저장한 뒤
"날짜 확인" 안내를 붙인다. 그래서 날짜는 되묻지 않고 말에 있을 때만 읽는다 (계획서 7-2장, C 요청).

읽는 표현
  - 오늘 · 금일 / 내일 · 명일 · 익일 / 모레 · 내일모레
  - 2026-11-05, 2026.11.05, 2026/11/05
  - 11월 5일            올해. 오늘보다 앞이면 내년
  - 5일에 · 5일 오후 · 5일 9시   이번 달. 오늘보다 앞이면 다음 달 ('3일간'처럼 기간은 읽지 않음)
읽지 않는 표현: 요일('금요일'), '다음 주', '11/5' (분수·번호와 헷갈림). 서로 다른 날짜가 둘 이상 나오면 읽지 않는다 (추측하지 않음).

기준일은 한국 시간이다. 서버(컨테이너)가 UTC여도 00~09시에 하루가 어긋나지 않게 고정 오프셋을 쓴다.
한국은 서머타임이 없어 고정 오프셋으로 충분하고, Windows에 tzdata가 없어도 동작한다.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

KST = timezone(timedelta(hours=9), name="Asia/Seoul")

_RELATIVE: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"내일\s*모레"), 2),
    (re.compile(r"모레"), 2),
    (re.compile(r"내일|명일|익일"), 1),
    (re.compile(r"오늘|금일"), 0),
)
_ISO_RE = re.compile(r"(?<!\d)(20\d{2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
# 'N일' 단독은 날짜일 때만: 뒤에 에·날·오전·오후·부터·N시가 와야 한다 ('3일간', '1일 2회'는 아님)
_DAY_ONLY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*일\s*(?=에|날|오전|오후|부터|\d{1,2}\s*시)")


def seoul_today() -> date:
    return datetime.now(KST).date()


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _next_month_day(today: date, day: int) -> date | None:
    """이번 달 day일. 오늘보다 앞이면 다음 달 day일. 이번 달에 없는 날('11월 31일')이면 None (다음 달로 넘기지 않음)."""
    found = _safe_date(today.year, today.month, day)
    if found is None:
        return None
    if found >= today:
        return found
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return _safe_date(year, month, day)


def find_work_dates(text: str, today: date) -> list[date]:
    """발화에 나온 날짜를 모두 돌려준다 (중복 제거, 나온 순서)."""
    found: list[tuple[int, date]] = []
    taken: list[tuple[int, int]] = []  # 이미 읽은 구간 ('11월 5일'의 '5일'을 다시 읽지 않게)

    def add(start: int, end: int, value: date | None) -> None:
        if value is None or any(s < end and start < e for s, e in taken):
            return
        taken.append((start, end))
        found.append((start, value))

    for m in _ISO_RE.finditer(text):
        add(m.start(), m.end(), _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in _MONTH_DAY_RE.finditer(text):
        value = _safe_date(today.year, int(m.group(1)), int(m.group(2)))
        if value is not None and value < today:
            value = _safe_date(today.year + 1, value.month, value.day)
        add(m.start(), m.end(), value)
    for m in _DAY_ONLY_RE.finditer(text):
        add(m.start(1), m.end(), _next_month_day(today, int(m.group(1))))
    for pattern, offset in _RELATIVE:
        for m in pattern.finditer(text):
            add(m.start(), m.end(), today + timedelta(days=offset))

    out: list[date] = []
    for _, value in sorted(found, key=lambda x: x[0]):
        if value not in out:
            out.append(value)
    return out


def extract_work_date(text: str, today: date | None = None) -> str | None:
    """날짜가 하나만 나오면 YYYY-MM-DD, 없거나 서로 다른 날짜가 둘 이상이면 None."""
    dates = find_work_dates(text, today or seoul_today())
    return dates[0].isoformat() if len(dates) == 1 else None
