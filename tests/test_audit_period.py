"""채팅 문장에서 감사 보고서 기간 읽기 (app/agent/audit_period.py). 기준일은 2026-11-19(목)로 고정한다."""

from datetime import date

import pytest

from app.agent.audit_period import read_audit_period

TODAY = date(2026, 11, 19)


def span(text: str, today: date = TODAY) -> tuple[str, str] | str:
    reading = read_audit_period(text, today)
    if not reading.ok:
        return reading.problem or "?"
    return reading.date_from.isoformat(), reading.date_to.isoformat()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("지난달 감사 보고서 만들어줘", ("2026-10-01", "2026-10-31")),
        ("지난 달 감사 자료", ("2026-10-01", "2026-10-31")),
        ("전월 감사 대응", ("2026-10-01", "2026-10-31")),
        ("이번 달 감사 보고서", ("2026-11-01", "2026-11-19")),  # 아직 끝나지 않은 달은 오늘까지
        ("지난주 감사 보고서", ("2026-11-09", "2026-11-15")),  # 월~일
        ("이번 주 감사 보고서", ("2026-11-16", "2026-11-19")),
        ("올해 감사 보고서", ("2026-01-01", "2026-11-19")),
        ("작년 감사 보고서", ("2025-01-01", "2025-12-31")),
        ("10월 감사 보고서", ("2026-10-01", "2026-10-31")),
        ("2월 감사 보고서", ("2026-02-01", "2026-02-28")),
        ("11월 감사 보고서", ("2026-11-01", "2026-11-19")),
        ("12월 감사 보고서", ("2025-12-01", "2025-12-31")),  # 오늘보다 뒤의 달은 작년
        ("2026년 9월 감사 보고서", ("2026-09-01", "2026-09-30")),
        ("2025년 12월 감사 보고서", ("2025-12-01", "2025-12-31")),
        ("2026-10-01 ~ 2026-10-31 감사 보고서", ("2026-10-01", "2026-10-31")),
        ("2026.10.31 부터 2026.10.01 까지", ("2026-10-01", "2026-10-31")),  # 순서가 뒤집혀도 구간으로
        ("2026/10/01~2026/10/15", ("2026-10-01", "2026-10-15")),
        ("10월 1일부터 10월 31일까지 감사 보고서", ("2026-10-01", "2026-10-31")),
        ("12월 1일부터 12월 31일까지", ("2025-12-01", "2025-12-31")),  # 시작일이 오늘보다 뒤면 작년
        ("2025년 10월 1일부터 2025년 10월 31일까지", ("2025-10-01", "2025-10-31")),  # 연도를 쓰면 그 연도
    ],
)
def test_period_read(text, expected):
    assert span(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "감사 보고서 만들어줘",
        "",
        "최근 3개월 감사 보고서",  # 읽지 않는 표현
        "다음 달 감사 보고서",
        "1분기 감사 보고서",
    ],
)
def test_period_missing(text):
    assert span(text) == "MISSING"


@pytest.mark.parametrize(
    "text",
    [
        "지난달과 이번 달 감사 보고서",  # 서로 다른 기간 둘
        "2026-10-01 감사 보고서",  # 날짜 하나
        "10월 31일 감사 보고서",
        "2026-10-01 2026-10-15 2026-10-31",  # 날짜 셋
        "2026-02-30 ~ 2026-03-10",  # 없는 날짜
        "2026-10-01 ~ 2026-10-31 지난달",  # 날짜 구간과 상대 표현이 섞임
        "9월 10월 감사 보고서",
    ],
)
def test_period_ambiguous(text):
    assert span(text) == "AMBIGUOUS"


@pytest.mark.parametrize("text", ["지난달 10월 감사 보고서", "2026년 10월 감사 보고서 지난달 기준", "지난달 감사 보고서 전월 기준"])
def test_same_period_written_twice_is_one_period(text):
    assert span(text) == ("2026-10-01", "2026-10-31")


def test_january_last_month_is_december_of_last_year():
    assert span("지난달 감사 보고서", date(2026, 1, 10)) == ("2025-12-01", "2025-12-31")


def test_last_week_across_year_boundary():
    # 2026-01-01은 목요일. 지난주 = 2025-12-22(월) ~ 12-28(일)
    assert span("지난주 감사 보고서", date(2026, 1, 1)) == ("2025-12-22", "2025-12-28")


def test_label_is_iso_range():
    assert read_audit_period("지난달", TODAY).label == "2026-10-01 ~ 2026-10-31"
    assert read_audit_period("감사", TODAY).label is None
