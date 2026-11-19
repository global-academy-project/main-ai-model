"""감사 대응 보고서 구성기의 상태값 기준.

**임시 기준이다(설계 13장 D11, 팀 결정 전).** 기준은 이 파일 한 곳에만 둔다.
팀이 정하면 이 파일의 함수만 고친다. 상태 이름은 평가어("위반", "부적정")를 쓰지 않는다.

- REVIEW    확인 필요: 기록 보완 여부를 사람이 살펴볼 항목
- REFERENCE 현황 참고: 사실 표기(판정 아님)
- OK        확인 필요 없음
- UNAVAILABLE 확인 불가: 조회하지 못했거나 값을 믿을 수 없음 (0건과 다르다)
"""

from __future__ import annotations

STATE_LABELS: dict[str, str] = {
    "REVIEW": "확인 필요",
    "REFERENCE": "현황 참고",
    "OK": "확인 필요 없음",
    "UNAVAILABLE": "확인 불가",
}


def risk_state(works: int, missing: int) -> str:
    """위험성평가: 평가 기록이 없는 작업이 1건 이상이면 확인 필요."""
    return "REVIEW" if missing > 0 else "OK"


def permit_state(permits: int) -> str:
    """작업허가 승인 현황: 승인·반려·차단은 사실 표기라 있으면 현황 참고."""
    return "REFERENCE" if permits > 0 else "OK"


def missing_measurement_state(count: int) -> str:
    """측정값 없이 승인된 허가: 1건이라도 있으면 확인 필요."""
    return "REVIEW" if count > 0 else "OK"


def blocked_state(count: int) -> str:
    """규칙 차단 기록: 시스템이 규칙에 따라 막은 기록이므로 현황 참고."""
    return "REFERENCE" if count > 0 else "OK"


def corrective_state(actions: int, open_count: int, overdue: int) -> str:
    """개선조치 이행: 기한이 지난 미완료가 있으면 확인 필요, 기한 안의 미완료는 현황 참고."""
    if overdue > 0:
        return "REVIEW"
    if open_count > 0:
        return "REFERENCE"
    return "OK"
