"""초안 품질 패치 A·B·D의 도우미 (drafting.py가 부른다). 새 필드를 만들지 않고, 응답 안의 값만 다듬는다.

A. TBM에 MSDS 응급조치 한 줄 (first_aid_points)
   - 실제 MSDS(publicdata/msds.py)의 first_aid는 "항목: 내용" 여러 줄이 이어진 긴 글이다. 첫 줄만 80자 안으로 줄여 쓴다.
   - 인화성 물질을 먼저, 최대 MAX_FIRST_AID_POINTS개. 기존 요점은 밀어내지 않는다 (남는 자리에만 넣는다, tbm_data 참고).
B. 허가서 category (flammable_category)
   - ghs_classes 첫 번째 대신 "인화성"이 들어간 분류를 먼저 쓴다. 없으면 예전처럼 첫 번째.
D. 허가서 방어 (classify_evaluation, block_reasons)
   - 규칙 판정 항목이 양식(MeasurementRequirement·AssignmentRequirement)에 맞지 않으면 그 칸만 건너뛰고 사유를 돌려준다.
     예전에는 KeyError나 fill_form의 FORM_INVALID로 실행 전체가 실패했다.
   - 승인 차단 사유(block_reasons)는 원래 판정 값으로만 계산하므로, 칸을 건너뛰어도 차단은 그대로다.

끄는 법: 아래 세 상수. 모두 끄면 예전 동작(A: 점 추가 없음, B: 첫 번째 분류, D: 예전 코드와 같은 값)이다. D는 끌 수 없는 방어라 상수가 없다.
  ADD_FIRST_AID_POINTS = False / PREFER_FLAMMABLE_CATEGORY = False
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import ValidationError

from app.schemas.agent_work import MsdsDto
from app.schemas.forms import AssignmentRequirement, MeasurementRequirement

logger = logging.getLogger(__name__)

ADD_FIRST_AID_POINTS = True
PREFER_FLAMMABLE_CATEGORY = True
MAX_FIRST_AID_POINTS = 2
FIRST_AID_MAX_CHARS = 80

Kind = Literal["measurement", "assignment", "check", "other"]


# ── B ───────────────────────────────────────────────────────

def flammable_category(ghs_classes: Sequence[str]) -> str | None:
    """허가서 category. "인화성"이 들어간 분류를 먼저, 없으면 첫 번째, 분류가 없으면 None."""
    if not PREFER_FLAMMABLE_CATEGORY:
        return ghs_classes[0] if ghs_classes else None  # 예전 동작 그대로
    classes = [c for c in ghs_classes if isinstance(c, str) and c.strip()]
    for c in classes:
        if "인화성" in c:
            return c
    return classes[0] if classes else None


# ── A ───────────────────────────────────────────────────────

def _shorten(text: str, limit: int = FIRST_AID_MAX_CHARS) -> str:
    """첫 줄만 쓰고 limit자 안으로 줄인다. 문장 끝(마침표)에서 자를 수 있으면 거기서 자른다."""
    line = next((" ".join(part.split()) for part in text.splitlines() if part.strip()), "")
    if len(line) <= limit:
        return line
    cut = line.rfind(".", 0, limit)
    if cut >= limit // 2:
        return line[: cut + 1]
    return line[: limit - 1].rstrip() + "…"


def first_aid_points(msds: Sequence[MsdsDto], *, limit: int = MAX_FIRST_AID_POINTS) -> list[str]:
    """TBM에 넣을 응급조치 한 줄들. 찾은 물질 중 first_aid가 있는 것만, 인화성 물질 먼저."""
    if not ADD_FIRST_AID_POINTS or limit <= 0:
        return []
    candidates = [m for m in msds if m.found and m.first_aid and m.first_aid.strip()]
    candidates.sort(key=lambda m: 0 if any("인화성" in g for g in m.ghs_classes) else 1)  # 정렬은 안정적이라 나머지 순서는 그대로
    points: list[str] = []
    for m in candidates:
        summary = _shorten(m.first_aid or "")
        if not summary:
            continue
        point = f"{m.name or m.query} 응급조치: {summary}"
        if point not in points:
            points.append(point)
        if len(points) == limit:
            break
    return points


# ── D ───────────────────────────────────────────────────────

def _problem(e: object, kind: str, why: str) -> tuple[Kind, None, str]:
    code = e.get("rule_code") if isinstance(e, Mapping) else None
    logger.warning("허가서 칸으로 옮기지 못한 판정 항목: kind=%s rule=%s why=%s", kind, code, why)
    return "other", None, str(code or "알 수 없음")


def classify_evaluation(
    e: object, role_labels: Mapping[str, str]
) -> tuple[Kind, dict[str, Any] | None, str | None]:
    """판정 항목 하나를 허가서 칸으로 옮긴다. (종류, 칸 값, 건너뛴 항목의 규칙 코드).

    건너뛴 항목만 세 번째 값이 채워진다. 유효한 항목은 예전 permit_data와 같은 값을 돌려준다.
    종류: measurement(측정값 칸) · assignment(배치 칸) · check(CHECK 상태의 안내 문구) · other(옮길 칸이 없음).
    """
    if not isinstance(e, Mapping):
        return _problem(e, "unknown", "항목이 객체가 아님")
    if "item" in e:
        entry = {
            "item": e.get("item"), "unit": e.get("unit"), "value": e.get("value"),
            "status": e.get("status"), "criterion": e.get("criterion"), "rule_code": e.get("rule_code"),
        }
        try:
            MeasurementRequirement.model_validate(entry)
        except ValidationError as exc:
            return _problem(e, "measurement", f"{exc.error_count()}개 값이 양식과 다름")
        return "measurement", entry, None
    if "role" in e:
        entry = {
            "role": e.get("role"), "label": role_labels.get(str(e.get("role"))), "assignee": e.get("assignee"),
            "status": "ASSIGNED" if e.get("assignee") else "REQUIRED",
        }
        try:
            AssignmentRequirement.model_validate(entry)
        except ValidationError as exc:
            return _problem(e, "assignment", f"{exc.error_count()}개 값이 양식과 다름")
        return "assignment", entry, None
    if e.get("status") == "CHECK":
        message = e.get("message")
        if isinstance(message, str) and message:
            return "check", {"message": message}, None
        return _problem(e, "check", "안내 문구(message)가 없음")
    return "other", None, None


def block_reasons(raw_evaluations: Sequence[object]) -> list[str]:
    """승인 차단 사유. 예전과 같은 기준(BLOCK이고 PASS가 아님)이고, message가 없으면 규칙 코드로 대신한다."""
    reasons: list[str] = []
    for e in raw_evaluations:
        if isinstance(e, Mapping) and e.get("severity") == "BLOCK" and e.get("status") != "PASS":
            message = e.get("message")
            reasons.append(message if isinstance(message, str) and message else f"규칙 {e.get('rule_code') or '?'} 확인이 필요합니다.")
        elif not isinstance(e, Mapping):
            reasons.append("규칙 판정 항목을 읽지 못했습니다. 확인 전에는 승인 요청을 할 수 없습니다.")
    return reasons
