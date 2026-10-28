"""classify_work 도구 (B, 1주차 규칙 기반 실제 구현).

작업 문장 → 작업유형 · 고위험 여부 · 필요한 작업허가.
  - 부정된 표현('크레인은 안 씀')은 빼고, 조건부('막히면 그라인더로 절단')는 넣는다 (안전 쪽).
  - 작업유형을 하나도 못 읽으면 high_risk=None(모름). '모름'을 '안전'으로 바꾸지 않는다.
  - 허가 종류는 코드가 정한다. 1개월 범위의 허가는 화기·밀폐공간·고소 3종(계획서 3장).
    전기·중량물은 고위험으로 표시하되 허가서는 만들지 않고 out_of_scope로 알린다 (양중·정전 허가는 4주차 확장).

나중에 태그 분류기(work_request 프로필)로 바꿀 때도 출력 모양(Classification)은 그대로 둔다.
"""

from __future__ import annotations
import re
from collections.abc import Mapping
from typing import Any
from app.schemas.agent_work import Classification, ClassificationReason
from .text_rules import is_negated, unique
from app.exceptions import ToolError

PERMIT_BY_WORK_TYPE: dict[str, str] = {
    "화기": "HOT_WORK_PERMIT",
    "밀폐공간": "CONFINED_SPACE_PERMIT",
    "고소": "HEIGHT_WORK_PERMIT",
}
OUT_OF_SCOPE_HIGH_RISK = ("전기", "중량물")

# 작업유형 → 판별 패턴 (임시 키워드. 정답 세트로 보완)
_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "화기": tuple(
        re.compile(p)
        for p in (r"(?<!소)화기", r"용접", r"용단", r"그라인더", r"토치", r"가스\s*절단", r"산소\s*절단", r"플라즈마", r"납땜", r"불꽃")
    ),
    "밀폐공간": tuple(
        re.compile(p)
        for p in (
            r"밀폐",
            r"맨홀",
            r"사일로",
            r"정화조",
            r"집수정",
            r"하수구",
            r"(?:탱크|반응기|저장조|피트|용기|보일러|덕트)\s*(?:내부|안|속)",
        )
    ),
    "고소": tuple(
        re.compile(p)
        for p in (
            r"고소",
            r"지붕",
            r"옥상",
            r"비계",
            r"사다리",
            r"스카이차",
            r"랙\s*위",
            r"철골\s*위",
            r"(?:높이|높은)\s*\d+(?:\.\d+)?\s*(?:m|미터)",
            r"\d+(?:\.\d+)?\s*(?:m|미터)\s*(?:이상\s*)?(?:높이|높은|위)",
        )
    ),
    "전기": tuple(
        re.compile(p)
        for p in (r"전기\s*(?:작업|공사)", r"정전\s*작업", r"배전반", r"분전반", r"활선", r"수배전", r"변압기", r"차단기\s*(?:교체|작업|점검)", r"결선")
    ),
    "중량물": tuple(
        re.compile(p) for p in (r"크레인", r"인양", r"호이스트", r"중량물", r"체인\s*블[록럭]", r"양중", r"지게차", r"줄걸이", r"리깅")
    ),
}
_GENERAL_RE = re.compile(r"교체|수리|점검|청소|설치|해체|정비|운반|도장|보수|분해|조립|작업")
_HEIGHT_MIN_M = 2.0
_HEIGHT_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|미터)")


def classify_text(text: str) -> Classification:
    reasons: list[ClassificationReason] = []
    excluded: list[str] = []
    for work_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            for m in pattern.finditer(text):
                keyword = m.group(0)
                if is_negated(text, m.end()):
                    excluded.append(f"{keyword} (부정)")
                    continue
                if work_type == "고소" and (hv := _HEIGHT_VALUE_RE.search(keyword)) and float(hv.group(1)) < _HEIGHT_MIN_M:
                    continue
                reasons.append(ClassificationReason(work_type=work_type, keyword=keyword))
    work_types = unique(r.work_type for r in reasons)
    if not work_types and _GENERAL_RE.search(text):
        work_types = ["일반"]

    high_risk: bool | None
    if not work_types:
        high_risk = None
    else:
        high_risk = any(w != "일반" for w in work_types)
    required = [PERMIT_BY_WORK_TYPE[w] for w in work_types if w in PERMIT_BY_WORK_TYPE]
    return Classification(
        work_types=work_types,
        high_risk=high_risk,
        required_permits=required,  # type: ignore[arg-type]
        can_start_before_approval=high_risk is False,
        reasons=_dedupe_reasons(reasons),
        excluded=unique(excluded),
        out_of_scope=[w for w in work_types if w in OUT_OF_SCOPE_HIGH_RISK],
    )


def _dedupe_reasons(reasons: list[ClassificationReason]) -> list[ClassificationReason]:
    seen: set[tuple[str, str]] = set()
    out: list[ClassificationReason] = []
    for r in reasons:
        if (r.work_type, r.keyword) not in seen:
            seen.add((r.work_type, r.keyword))
            out.append(r)
    return out


CLASSIFY_WORK_PARAMETERS = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "작업 내용 원문"}},
    "required": ["text"],
}


def classify_work(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """도구 레지스트리용 진입점."""
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("TOOL_INPUT_INVALID", "classify_work에는 작업 내용(text)이 필요합니다.", tool="classify_work")
    return classify_text(text).model_dump()
