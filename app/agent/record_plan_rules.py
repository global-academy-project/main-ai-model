"""기록 질문 계획 단계의 규칙 (B, LLM 없음).

D의 `RecordQueryAgent._plan`은 질문을 모델에 주고 기록 유형·기간 JSON을 받는다. 이 모듈은 **분명한 질문만** 코드로 먼저
처리한다. 처리하지 못하면 None을 돌려주고, 호출하는 쪽이 D의 기존 LLM 계획으로 넘긴다 (되묻기도 거기서 한다).

규칙으로 정하는 조건 (전부 만족해야 한다)
  1. 기록 유형이 하나로 정해진다 (키워드 규칙, 아래 TYPE_RULES). 둘 이상이거나 없으면 None.
     제외어로 빠진 유형이 같이 보이면(예: 'TBM 누락 + 허가 승인 현황') 두 가지를 묻는 것일 수 있어 None.
  2. 기간이 하나로 읽힌다 (`audit_period.read_audit_period`). 못 읽거나(MISSING) 여럿(AMBIGUOUS)이면 None.
     ('최근 3개월' 같은 표현은 이 읽기기가 안 읽으므로 모델 쪽에서 처리하게 둔다.)
  3. 유형에 필요한 추가 조건을 코드로 채울 수 있다. 값을 모르는 조건이 질문에 보이면 None.
  4. 기간이 366일을 넘지 않는다.
이전 대화(context)를 이어 묻는 문장은 유형 키워드가 없으므로 자연히 None이 된다.

임시값: TYPE_RULES 키워드, 허가·작업 종류 코드(HOT·CONFINED·HEIGHT), UNSUPPORTED_HINTS는 내 휴리스틱이다.
Spring `search_records`의 조건 값 목록과 대조해 확정한다. 끄기: 설정 RECORD_PLAN_RULES=false (기본).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from .audit_period import read_audit_period

MAX_DAYS = 365  # D의 _valid_params와 같다 (끝 - 시작)


@dataclass(frozen=True)
class RulePlan:
    query_type: str
    params: dict[str, Any]


# 유형별 키워드: 모든 묶음에서 하나 이상 있어야 하고(AND of OR), 제외어가 있으면 그 유형이 아니다. 공백을 뺀 소문자 문장에 건다.
TYPE_RULES: dict[str, dict[str, tuple[tuple[str, ...], ...]]] = {
    "missing_measurements": {
        "all": (("가스측정", "측정값", "측정"), ("누락", "빠진", "빠짐", "없는", "없이", "미기록", "미입력")),
        "not": (),
    },
    "permit_approval_statistics": {
        "all": (("허가",), ("승인", "반려"), ("통계", "현황", "건수", "몇건", "승인률", "승인율", "비율")),
        "not": (("대기", "누락", "측정", "차단", "승인자"),),
    },
    "pending_approvals": {
        "all": (("승인", "결재"), ("대기", "기다리", "미승인", "미결")),
        "not": (),
    },
    "rejected_revisions": {
        "all": (("반려",), ("수정", "재작성", "재제출", "고친", "보완")),
        "not": (),
    },
    "open_corrective_actions": {
        "all": (("시정조치", "개선조치", "지적사항", "조치"), ("미완료", "안끝난", "남은", "열려", "진행중", "미처리")),
        "not": (("조치율",),),
    },
    "failed_inspections": {
        "all": (("점검",), ("부적합", "불합격", "불량", "실패", "미흡")),
        "not": (),
    },
    "missing_tbm": {
        "all": (("tbm", "티비엠", "작업전대화"), ("누락", "빠진", "빠짐", "안한", "미실시", "안했", "없는")),
        "not": (),
    },
    "risk_assessment_rate": {
        "all": (("위험성평가",), ("실시", "현황", "비율", "얼마나", "몇건", "건수")),
        "not": (),
    },
    "blocked_records": {
        "all": (("차단",), ("기록", "허가", "내역", "현황", "목록", "건수", "몇건")),
        "not": (),
    },
    # approver_statistics, record_change_history는 승인자·기록 번호가 필요해 규칙으로 정하지 않는다 (항상 None → LLM).
}

# 허가·작업 종류 (임시). 한 종류만 보일 때만 조건으로 넣고, 둘 이상이면 None.
_KIND_WORDS: dict[str, tuple[str, ...]] = {
    "HOT": ("화기", "용접", "불꽃"),
    "CONFINED": ("밀폐", "질식"),
    "HEIGHT": ("고소", "비계", "추락"),
}
_PERMIT_TYPE_QUERIES = frozenset({"missing_measurements", "permit_approval_statistics", "blocked_records"})
_WORK_TYPE_QUERIES = frozenset({"risk_assessment_rate", "missing_tbm"})

# 값을 모르는 조건이 보이면 규칙을 포기한다 (모델이 처리). 키는 유형이 아니라 포기 사유별 단어다.
UNSUPPORTED_HINTS: dict[str, tuple[str, ...]] = {
    "pending_approvals": ("허가", "위험성평가", "tbm", "점검", "일지", "단계", "1차", "2차", "3차"),
    "rejected_revisions": ("허가", "위험성평가", "tbm", "점검", "일지"),
    "open_corrective_actions": ("기한", "지연", "초과", "완료", "검증", "취소"),
    "failed_inspections": ("구역", "존", "템플릿", "양식", "호기"),
}
_DIGIT = re.compile(r"\d")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _scan(message: str) -> tuple[list[str], list[str]]:
    """(키워드가 맞은 유형, 제외어 때문에 빠진 유형). 정렬은 TYPE_RULES 순서."""
    text = _compact(message)
    found: list[str] = []
    vetoed: list[str] = []
    for query_type, rule in TYPE_RULES.items():
        if not all(any(word in text for word in group) for group in rule["all"]):
            continue
        if any(any(word in text for word in group) for group in rule["not"]):
            vetoed.append(query_type)
            continue
        found.append(query_type)
    return found, vetoed


def match_types(message: str) -> list[str]:
    """키워드 규칙에 걸리는 기록 유형 목록 (제외어로 빠진 유형은 넣지 않는다)."""
    return _scan(message)[0]


def _kind(text: str) -> tuple[str | None, bool]:
    """(종류 코드, 둘 이상이라 정하지 못함)."""
    hits = [code for code, words in _KIND_WORDS.items() if any(word in text for word in words)]
    if len(hits) > 1:
        return None, True
    return (hits[0] if hits else None), False


def _extra_params(query_type: str, message: str) -> dict[str, Any] | None:
    """유형별 추가 조건. 채울 수 없는 조건이 보이면 None."""
    text = _compact(message)
    params: dict[str, Any] = {}
    if query_type in _PERMIT_TYPE_QUERIES or query_type in _WORK_TYPE_QUERIES:
        code, mixed = _kind(text)
        if mixed:
            return None
        if code is not None:
            params["permitType" if query_type in _PERMIT_TYPE_QUERIES else "workType"] = code
        return params
    hints = UNSUPPORTED_HINTS.get(query_type, ())
    if any(word in text.replace("미완료", "") for word in hints):  # '미완료'는 이 유형의 키워드라 '완료' 힌트에서 뺀다
        return None
    if query_type == "failed_inspections" and _DIGIT.search(_without_dates(message)):
        return None  # 구역·양식 번호일 수 있다
    return params


def _without_dates(message: str) -> str:
    """기간 표현의 숫자(10월, 2026-10-01 …)를 지워 구역·번호 숫자만 남긴다."""
    text = re.sub(r"20\d{2}\s*[-./년]\s*\d{1,2}(\s*[-./월]\s*\d{1,2}\s*일?)?", " ", message)
    return re.sub(r"\d{1,2}\s*월(\s*\d{1,2}\s*일)?", " ", text)


def rule_plan(message: str, context: Mapping[str, Any] | None = None, today: date | None = None) -> RulePlan | None:
    """규칙으로 정할 수 있으면 RulePlan, 아니면 None (모델 계획으로 넘긴다)."""
    del context  # 이어 묻기는 유형 키워드가 없어 여기까지 오지 않는다. 있는 문장은 그 자체로 완결이다.
    types, vetoed = _scan(message)
    if len(types) != 1 or vetoed:
        return None  # 유형이 하나로 안 정해지거나, 제외어로 빠진 다른 유형이 같이 보이면 두 가지를 묻는 질문일 수 있다
    query_type = types[0]
    reading = read_audit_period(message, today)
    if not reading.ok:
        return None
    assert reading.date_from is not None and reading.date_to is not None
    if reading.date_from > reading.date_to or (reading.date_to - reading.date_from).days > MAX_DAYS:
        return None
    extra = _extra_params(query_type, message)
    if extra is None:
        return None
    params: dict[str, Any] = {"from": reading.date_from.isoformat(), "to": reading.date_to.isoformat(), **extra}
    return RulePlan(query_type=query_type, params=params)
