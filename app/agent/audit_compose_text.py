"""구성기의 한글 표기·항목 사양·서술문 틀.

화면에 내부 필드명·영어 코드를 보이지 않는다. 모르는 코드값은 영어 그대로 내지 않고 일반 표현으로 바꾼다.
문구는 제안이며 팀 결정 전에는 확정이 아니다. 서술문은 모두 "~임/~함" 보고서체이고 평가어를 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.audit_compose import ItemBasis

LEGAL = "LEGAL"
INTERNAL = "INTERNAL"

# 아차사고는 조회 항목이 아직 없어 이름만 둔다(설계 13장 D9). 실제 queryType이 생기면 바꾼다.
NEAR_MISS_RECEIPT = "near_miss_receipt"
NEAR_MISS_LINK = "near_miss_action_link"


@dataclass(frozen=True)
class ItemSpec:
    query_type: str
    area: str
    kind: str
    basis: tuple[ItemBasis, ...]
    point: str
    summary_label: str


SPECS: dict[str, ItemSpec] = {
    "risk_assessment_rate": ItemSpec(
        "risk_assessment_rate",
        "위험성평가 실시",
        LEGAL,
        (ItemBasis(title="위험성평가 실시", legal_basis="산업안전보건법 제36조", kind=LEGAL),),
        "작업별 위험성평가 기록의 작성·승인 여부",
        "평가 기록이 없는 작업 / 전체 작업",
    ),
    "permit_approval_statistics": ItemSpec(
        "permit_approval_statistics",
        "작업허가 승인 현황",
        INTERNAL,
        (ItemBasis(title="작업허가 승인 절차", legal_basis=None, kind=INTERNAL),),
        "작업허가의 승인 요청·승인·반려·차단 현황",
        "승인 / (승인+반려)",
    ),
    NEAR_MISS_RECEIPT: ItemSpec(
        NEAR_MISS_RECEIPT,
        "아차사고 접수 현황",
        INTERNAL,
        (ItemBasis(title="아차사고 접수", legal_basis=None, kind=INTERNAL),),
        "아차사고 접수 현황(발생 형태·작업 유형·구역)",
        "-",
    ),
    NEAR_MISS_LINK: ItemSpec(
        NEAR_MISS_LINK,
        "아차사고 조치 연결 현황",
        INTERNAL,
        (ItemBasis(title="아차사고 조치 연결", legal_basis=None, kind=INTERNAL),),
        "아차사고의 조치 연결 현황(개선조치·위험성평가 갱신·작업허가 경고)",
        "-",
    ),
    "missing_measurements": ItemSpec(
        "missing_measurements",
        "측정값 없이 승인된 허가",
        INTERNAL,
        (ItemBasis(title="작업 전 측정값 확인", legal_basis=None, kind=INTERNAL),),
        "승인된 허가 중 작업 전 측정값이 없는 허가",
        "측정값 없이 승인된 허가",
    ),
    "blocked_records": ItemSpec(
        "blocked_records",
        "규칙 차단 기록",
        INTERNAL,
        (ItemBasis(title="규칙 판정 차단 기록", legal_basis=None, kind=INTERNAL),),
        "규칙 판정으로 차단된 허가 기록",
        "차단 기록",
    ),
    "corrective_action_rate": ItemSpec(
        "corrective_action_rate",
        "개선조치 이행",
        INTERNAL,
        (ItemBasis(title="개선조치 이행", legal_basis=None, kind=INTERNAL),),
        "개선조치의 완료·검증·기한 경과 현황",
        "완료(검증 포함) / 등록 개선조치",
    ),
}

# (묶음 번호, 묶음 제목, 세부 점검 queryType 순서)
GROUPS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "위험성평가·작업허가", ("risk_assessment_rate", "permit_approval_statistics")),
    (2, "아차사고 내역", (NEAR_MISS_RECEIPT, NEAR_MISS_LINK)),
    (3, "그 밖의 안전관리 기록", ("missing_measurements", "blocked_records", "corrective_action_rate")),
)

NOT_SUPPORTED_SENTENCE = "조회 항목이 아직 없어 이번 보고서에서는 확인하지 못함."

# ---------------------------------------------------------------- 코드값 → 한글(모르는 값은 일반 표현)

_WORK_TYPE_KO = {"HOT": "화기작업", "CONFINED": "밀폐공간작업", "HEIGHT": "고소작업"}
_PERMIT_TYPE_KO = {"HOT": "화기작업 허가", "CONFINED": "밀폐공간작업 허가", "HEIGHT": "고소작업 허가"}
_ACTION_STATUS_KO = {"DONE": "완료", "VERIFIED": "검증 완료", "CANCELED": "취소"}
_ACTION_SOURCE_KO = {
    "NEAR_MISS": "아차사고",
    "INSPECTION": "점검",
    "RULE_VIOLATION": "규칙 판정",
    "AUDIT": "감사",
    "OTHER": "기타",
}


def work_type_ko(code: Any) -> str:
    return _WORK_TYPE_KO.get(code, "기타 작업") if isinstance(code, str) else "기타 작업"


def permit_type_ko(code: Any) -> str:
    return _PERMIT_TYPE_KO.get(code, "작업허가") if isinstance(code, str) else "작업허가"


def action_status_ko(code: Any) -> str:
    # Spring은 완료·검증 완료가 아니면 미완료로 센다(open = 등록 − 완료). 같은 기준으로 표기한다.
    return _ACTION_STATUS_KO.get(code, "미완료") if isinstance(code, str) else "미완료"


def action_source_ko(code: Any) -> str:
    return _ACTION_SOURCE_KO.get(code, "기타") if isinstance(code, str) else "기타"


# ---------------------------------------------------------------- 기록 줄 재료


def clip(text: Any, limit: int) -> str | None:
    """자유 텍스트는 짧게 줄여 쓴다(개인정보·길이 규칙). 문자열이 아니거나 비면 None."""
    if not isinstance(text, str):
        return None
    flat = " ".join(text.split())
    if not flat:
        return None
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def date_part(value: Any) -> str | None:
    """'2026-10-14T09:30:00' 같은 값에서 날짜만. 형식이 다르면 None."""
    if isinstance(value, str) and len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:10]
    return None


def join_parts(parts: list[str | None]) -> str:
    return " · ".join(p for p in parts if p)


# ---------------------------------------------------------------- 서술문 틀 (숫자는 호출하는 쪽이 코드로 계산해 넘긴다)


_BASIS_SENTENCES: dict[str, str] = {
    "risk_assessment_rate": "위험성평가는 산업안전보건법 제36조에 따른 법정 의무 사항으로, 작업별 작성·승인 기록을 확인함.",
    "permit_approval_statistics": "작업허가는 사내 절차로, 승인 요청·승인·반려·차단 기록을 확인함.",
    "missing_measurements": "작업 전 측정값 확인은 사내 절차로, 승인된 허가의 측정값 기록 여부를 확인함.",
    "blocked_records": "규칙 판정 차단은 사내 절차로, 차단된 허가 기록을 확인함.",
    "corrective_action_rate": "개선조치 이행은 사내 절차로, 등록된 조치의 완료·검증 현황을 확인함.",
}


def basis_sentence(spec: ItemSpec) -> str:
    return _BASIS_SENTENCES[spec.query_type]


def truncated_sentence(total: int, shown: int) -> str:
    return f"전체 {total}건 중 {shown}건만 표시함."
