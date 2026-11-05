"""문장 → 규칙 입력 추출 (C 담당): LLM 출력(JSON 객체) → 검사한 추출 결과. 표준 라이브러리만 쓴다 (모델 없이 테스트).

버리는 값 (원칙: LLM 은 숫자를 추정하지 않는다, 확실하지 않으면 비운다)
  - 요청에 없는 항목·작업 종류, 모르는 단위
  - 근거 문구(evidence)가 원문에 그대로 없거나, 근거 문구 안에 값 숫자가 그대로 없음
  - 음수, % 단위인데 100 초과
  - 같은 항목이 두 번 (값이 같으면 하나만, 다르면 둘 다 버림)
  - 감시자 이름이 원문에 없음
Spring 도 같은 검사를 한 번 더 한다 (ExtractionMerger). 여기서도 거르는 것은 로그·평가를 깨끗하게 두려고.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

UNITS: frozenset[str] = frozenset({"PCT", "PPM", "PCT_LEL"})
# 항목마다 받을 단위 (Spring MeasurementItem·사전과 같음). LEL 은 "5%"도 %LEL 로 본다
ITEM_UNITS: dict[str, frozenset[str]] = {
    "O2": frozenset({"PCT"}),
    "CO2": frozenset({"PCT"}),
    "LEL": frozenset({"PCT_LEL", "PCT"}),
    "H2S": frozenset({"PPM"}),
    "CO": frozenset({"PPM"}),
    "NH3": frozenset({"PPM"}),
    "CL2": frozenset({"PPM"}),
}
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
MAX_NAME_LENGTH = 20


@dataclass(frozen=True)
class Measurement:
    item: str
    value: Decimal
    unit: str
    evidence: str


@dataclass(frozen=True)
class WorkTypeHit:
    type: str
    evidence: str


@dataclass
class ExtractedRuleInput:
    measurements: list[Measurement] = field(default_factory=list)
    work_types: list[WorkTypeHit] = field(default_factory=list)
    watcher_name: str | None = None
    watcher_evidence: str | None = None
    dropped: list[str] = field(default_factory=list)   # 버린 이유 (로그·평가용, 응답에는 안 나감)


def parse_rule_input(raw: dict[str, Any] | None, text: str, items: list[str], work_types: list[str]) -> ExtractedRuleInput:
    """raw: 모델이 낸 JSON 객체 (없으면 빈 결과). text: 원문."""
    out = ExtractedRuleInput()
    if not isinstance(raw, dict):
        return out
    allowed_items = {i.upper() for i in items} & ITEM_UNITS.keys()
    allowed_types = {t.upper() for t in work_types}

    seen: dict[str, Measurement] = {}
    conflict: set[str] = set()
    for m in _list(raw.get("measurements")):
        item = _str(m.get("item")).upper()
        unit = _str(m.get("unit")).upper()
        evidence = _str(m.get("evidence"))
        value = _decimal(m.get("value"))
        if item not in allowed_items:
            out.dropped.append(f"모르는 항목: {item!r}")
            continue
        if value is None or value < 0 or (unit in ("PCT", "PCT_LEL") and value > 100):
            out.dropped.append(f"{item}: 값이 숫자가 아니거나 범위 밖")
            continue
        if unit not in UNITS or unit not in ITEM_UNITS[item]:
            out.dropped.append(f"{item}: 단위 {unit!r} 를 받지 않음")
            continue
        if not evidence or evidence not in text or not _contains_number(evidence, value):
            out.dropped.append(f"{item}: 근거 문구가 원문에 없거나 값이 근거에 없음")
            continue
        prev = seen.get(item)
        if prev is not None and prev.value != value:
            conflict.add(item)
            continue
        seen.setdefault(item, Measurement(item, value, unit, evidence))
    for item in conflict:
        seen.pop(item, None)
        out.dropped.append(f"{item}: 값이 둘 이상")
    out.measurements = list(seen.values())

    types: dict[str, WorkTypeHit] = {}
    for t in _list(raw.get("workTypes")):
        kind = _str(t.get("type")).upper()
        evidence = _str(t.get("evidence"))
        if kind in allowed_types and evidence and evidence in text:
            types.setdefault(kind, WorkTypeHit(kind, evidence))
    out.work_types = list(types.values())

    name = _str(raw.get("watcherName"))
    if 2 <= len(name) <= MAX_NAME_LENGTH and name in text:
        out.watcher_name = name
        evidence = _str(raw.get("watcherEvidence"))
        out.watcher_evidence = evidence if evidence and evidence in text else None
    return out


def _contains_number(evidence: str, value: Decimal) -> bool:
    return any(Decimal(n) == value for n in _NUMBER_RE.findall(evidence))


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        d = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return d if d.is_finite() else None
