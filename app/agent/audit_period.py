"""채팅 문장에서 감사 대응 보고서의 기간(dateFrom ~ dateTo)을 읽는다.

채팅에서 E(감사 대응)로 분류된 말이 어느 기간의 보고서를 원하는지 정한다. 모델을 부르지 않는다.
읽은 기간은 Spring이 감사 보고서를 만들 때 그대로 쓰므로, **추측하지 않는다**: 없으면 MISSING, 둘 이상으로 읽히면
AMBIGUOUS로 돌려주고 호출하는 쪽이 되묻는다.

읽는 표현 (기준일은 한국 시간, `work_date.seoul_today`)
  - 지난달 · 지난 달 · 전월 · 저번 달      지난달 1일 ~ 말일
  - 이번 달 · 금월 · 당월                   이번 달 1일 ~ 오늘
  - 지난주 · 지난 주 · 저번 주              지난주 월요일 ~ 일요일
  - 이번 주                                 이번 주 월요일 ~ 오늘
  - 올해 · 금년                             1월 1일 ~ 오늘
  - 작년 · 지난해 · 전년                    작년 1월 1일 ~ 12월 31일
  - 10월 · 2026년 10월                      그 달 1일 ~ 말일 (이번 달이면 오늘까지). '10월'은 올해, 오늘보다 뒤의 달이면 작년
  - 2026-10-01 ~ 2026-10-31 (날짜 둘)       그 구간. 구분자 - . / 와 '2026년 10월 1일' 꼴도 받는다
  - 10월 1일부터 10월 31일까지 (날짜 둘)    그 구간. 연도는 올해, 시작일이 오늘보다 뒤면 작년
읽지 않는 표현: 날짜 하나만, 서로 다른 기간이 둘 이상('지난달과 이번 달'), '최근 3개월', '1분기', '다음 달'.
  → 날짜 하나만 / 둘 이상이면 AMBIGUOUS, 아무것도 없거나 읽지 않는 표현이면 MISSING.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from .work_date import seoul_today

Problem = Literal["MISSING", "AMBIGUOUS"]


@dataclass(frozen=True)
class AuditPeriodReading:
    date_from: date | None = None
    date_to: date | None = None
    problem: Problem | None = None  # None이면 읽었다

    @property
    def ok(self) -> bool:
        return self.problem is None and self.date_from is not None and self.date_to is not None

    @property
    def label(self) -> str | None:
        """화면 안내용 'YYYY-MM-DD ~ YYYY-MM-DD'."""
        if not self.ok:
            return None
        return f"{self.date_from.isoformat()} ~ {self.date_to.isoformat()}"  # type: ignore[union-attr]


_ISO_RE = re.compile(r"(?<!\d)(20\d{2})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_YMD_KO_RE = re.compile(r"(?<!\d)(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_YEAR_MONTH_RE = re.compile(r"(?<!\d)(20\d{2})\s*년\s*(\d{1,2})\s*월(?!\s*\d{1,2}\s*일)")
_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월")

_LAST_MONTH_RE = re.compile(r"지난\s*달|전월|저번\s*달")
_THIS_MONTH_RE = re.compile(r"이번\s*달|금월|당월")
_LAST_WEEK_RE = re.compile(r"지난\s*주|저번\s*주")
_THIS_WEEK_RE = re.compile(r"이번\s*주")
_THIS_YEAR_RE = re.compile(r"올해|금년")
_LAST_YEAR_RE = re.compile(r"작년|지난해|전년")

Span = tuple[date, date]


def _blank(pattern: re.Pattern[str], text: str) -> tuple[list[re.Match[str]], str]:
    """패턴에 걸린 곳을 지워(공백으로) 다음 패턴이 같은 글자를 다시 읽지 않게 한다."""
    found = list(pattern.finditer(text))
    for m in reversed(found):
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return found, text


def _month_span(year: int, month: int, today: date) -> Span:
    first = date(year, month, 1)
    last = date(year, month, monthrange(year, month)[1])
    if (year, month) == (today.year, today.month):
        last = today  # 이번 달은 아직 끝나지 않았다
    return first, last


def _previous_month(today: date) -> Span:
    year, month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
    return _month_span(year, month, today)


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def read_audit_period(text: str, today: date | None = None) -> AuditPeriodReading:
    """문장에서 기간 하나를 읽는다. 못 읽으면(MISSING)·여럿이면(AMBIGUOUS) 사유만 돌려준다."""
    today = today or seoul_today()
    rest = text or ""

    # 1) 날짜를 직접 쓴 경우: 2026-10-01, 10월 1일. 정확히 둘이어야 구간이다.
    iso, rest = _blank(_ISO_RE, rest)
    iso_ko, rest = _blank(_YMD_KO_RE, rest)  # 2026년 10월 1일 (연도를 쓴 날짜는 연도를 그대로 쓴다)
    iso = iso + iso_ko
    ym, rest = _blank(_YEAR_MONTH_RE, rest)
    md, rest = _blank(_MONTH_DAY_RE, rest)

    explicit: list[date] = []
    for m in iso:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d is None:
            return AuditPeriodReading(problem="AMBIGUOUS")
        explicit.append(d)
    md_dates: list[tuple[int, int]] = [(int(m.group(1)), int(m.group(2))) for m in md]

    spans: list[Span] = []

    for m in ym:
        month = int(m.group(2))
        if not 1 <= month <= 12:
            return AuditPeriodReading(problem="AMBIGUOUS")
        spans.append(_month_span(int(m.group(1)), month, today))

    # 월·일로 쓴 날짜는 올해로 읽고, 가장 이른 날이 오늘보다 뒤면 작년으로 읽는다 (연도를 쓴 날짜와는 섞지 않는다)
    if md_dates:
        if explicit:
            return AuditPeriodReading(problem="AMBIGUOUS")
        year = today.year
        first = [_safe_date(year, mo, d) for mo, d in md_dates]
        if any(v is None for v in first):
            return AuditPeriodReading(problem="AMBIGUOUS")
        if min(v for v in first if v is not None) > today:
            year -= 1
        explicit = [v for mo, d in md_dates if (v := _safe_date(year, mo, d)) is not None]
        if len(explicit) != len(md_dates):
            return AuditPeriodReading(problem="AMBIGUOUS")

    # 2) 상대 표현과 'N월'. 날짜 구간과 섞이면 어느 쪽인지 알 수 없으므로 아래에서 AMBIGUOUS로 둔다
    for pattern, make in (
        (_LAST_MONTH_RE, lambda: _previous_month(today)),
        (_THIS_MONTH_RE, lambda: _month_span(today.year, today.month, today)),
        (_LAST_WEEK_RE, lambda: (_monday(today) - timedelta(days=7), _monday(today) - timedelta(days=1))),
        (_THIS_WEEK_RE, lambda: (_monday(today), today)),
        (_THIS_YEAR_RE, lambda: (date(today.year, 1, 1), today)),
        (_LAST_YEAR_RE, lambda: (date(today.year - 1, 1, 1), date(today.year - 1, 12, 31))),
    ):
        found, rest = _blank(pattern, rest)
        if found:
            spans.append(make())

    months, rest = _blank(_MONTH_RE, rest)
    for m in months:
        month = int(m.group(1))
        if not 1 <= month <= 12:
            return AuditPeriodReading(problem="AMBIGUOUS")
        spans.append(_month_span(today.year - 1 if month > today.month else today.year, month, today))

    if explicit:
        if len(explicit) != 2 or spans:
            return AuditPeriodReading(problem="AMBIGUOUS")  # 날짜 하나, 셋 이상, 다른 기간과 섞임
        start, end = sorted(explicit)
        return AuditPeriodReading(start, end)

    distinct = list(dict.fromkeys(spans))
    if not distinct:
        return AuditPeriodReading(problem="MISSING")
    if len(distinct) > 1:
        return AuditPeriodReading(problem="AMBIGUOUS")
    return AuditPeriodReading(*distinct[0])
