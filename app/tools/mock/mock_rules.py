"""[연습용 mock] check_rules (C 담당 규칙 엔진의 자리 표시).

입력
  {"permit_types": ["HOT_WORK_PERMIT", ...],
   "measurements": {"O2": 20.9, "LEL": 0, ...},          측정값 (없으면 생략)
   "assignments": {"FIRE_WATCHER": "이름" | null, "ATTENDANT": ...},
   "location": str | null, "period": {"start", "end"}}
출력
  {"blocked": bool,
   "evaluations": [{"rule_code", "title", "status": PASS|FAIL|MISSING|CHECK, "severity": BLOCK|WARN,
                    "message", "basis", "item", "unit", "value", "criterion"}],
   "conflicts": [str], "notes": [str]}

blocked = severity가 BLOCK인 평가 중 PASS가 아닌 것이 하나라도 있음 → 승인 요청 차단 (계획서 3장 "측정값 없으면 승인 요청 차단").

근거 조항: 밀폐공간 적정공기 4개는 업무 DB 규칙 시드(R__rules_seed.sql)에 원문 확인된 산안규칙 제618조를 쓴다.
화기·고소 규칙의 기준값과 조항은 C가 원문을 확인하기 전이라 '확인 필요'로 표시한다.
동시작업 충돌은 업무 기록 조회가 필요해서 mock은 검사하지 않는다 (notes에 남긴다).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from app.exceptions import ToolError

_ART_618 = "산업안전보건기준에 관한 규칙 제618조 (적정공기)"
_UNCONFIRMED = "[mock] 근거 조항 확인 필요 (C 규칙 엔진에서 확정)"

@dataclass(frozen=True)
class _MeasureRule:
    code: str
    title: str
    item: str
    unit: str
    criterion: str
    ok: Callable[[float], bool]
    basis: str


@dataclass(frozen=True)
class _AssignRule:
    code: str
    title: str
    role: str
    basis: str


@dataclass(frozen=True)
class _CheckRule:
    code: str
    title: str
    message: str
    basis: str


_RULES: dict[str, tuple[Any, ...]] = {
    "HOT_WORK_PERMIT": (
        _MeasureRule("HOT.LEL", "가연성가스 농도 측정", "LEL", "PCT_LEL", "[mock] LEL 10% 미만", lambda v: v < 10, _UNCONFIRMED),
        _AssignRule("HOT.FIRE_WATCHER", "화재감시자 지정", "FIRE_WATCHER", _UNCONFIRMED),
        _CheckRule("HOT.EXTINGUISHER", "소화기 비치", "작업 장소에 소화기를 비치했는지 현장에서 확인하세요.", _UNCONFIRMED),
    ),
    "CONFINED_SPACE_PERMIT": (
        _MeasureRule("CS.O2", "산소 농도", "O2", "PCT", "18% 이상 23.5% 미만", lambda v: 18 <= v < 23.5, _ART_618),
        _MeasureRule("CS.CO2", "탄산가스 농도", "CO2", "PCT", "1.5% 미만", lambda v: v < 1.5, _ART_618),
        _MeasureRule("CS.CO", "일산화탄소 농도", "CO", "PPM", "30ppm 미만", lambda v: v < 30, _ART_618),
        _MeasureRule("CS.H2S", "황화수소 농도", "H2S", "PPM", "10ppm 미만", lambda v: v < 10, _ART_618),
        _AssignRule("CS.ATTENDANT", "감시인 지정", "ATTENDANT", _UNCONFIRMED),
    ),
    "HEIGHT_WORK_PERMIT": (
        _CheckRule("HT.HARNESS", "안전대 부착설비", "안전대 부착설비가 있는지 현장에서 확인하세요.", _UNCONFIRMED),
        _CheckRule("HT.PLATFORM", "작업발판·안전난간", "작업발판과 안전난간 설치 여부를 확인하세요.", _UNCONFIRMED),
    ),
}
_ROLE_LABELS = {"FIRE_WATCHER": "화재감시자", "ATTENDANT": "감시인"}


def check_rules(arguments: Mapping[str, Any]) -> dict[str, Any]:
    permit_types = arguments.get("permit_types", [])
    measurements = arguments.get("measurements") or {}
    assignments = arguments.get("assignments") or {}
    if not isinstance(permit_types, list) or not isinstance(measurements, Mapping) or not isinstance(assignments, Mapping):
        raise ToolError("TOOL_INPUT_INVALID", "check_rules 입력 형식이 맞지 않습니다.", tool="check_rules")
    unknown = [p for p in permit_types if p not in _RULES]
    if unknown:
        raise ToolError("TOOL_INPUT_INVALID", "check_rules가 모르는 허가 종류가 있습니다.", tool="check_rules")

    evaluations: list[dict[str, Any]] = []
    for permit in permit_types:
        for rule in _RULES[permit]:
            if isinstance(rule, _MeasureRule):
                value = measurements.get(rule.item)
                if value is None:
                    status, message = "MISSING", f"{rule.title} 값이 없습니다. 측정 전에는 승인 요청을 할 수 없습니다."
                elif rule.ok(float(value)):
                    status, message = "PASS", f"{rule.title} {value} — 기준({rule.criterion}) 충족."
                else:
                    status, message = "FAIL", f"{rule.title} {value} — 기준({rule.criterion}) 미달. 환기 후 다시 측정하세요."
                evaluations.append({
                    "rule_code": rule.code, "title": rule.title, "status": status, "severity": "BLOCK",
                    "message": message, "basis": rule.basis, "item": rule.item, "unit": rule.unit,
                    "value": value, "criterion": rule.criterion,
                })
            elif isinstance(rule, _AssignRule):
                assignee = assignments.get(rule.role)
                label = _ROLE_LABELS[rule.role]
                evaluations.append({
                    "rule_code": rule.code, "title": rule.title, "status": "PASS" if assignee else "MISSING",
                    "severity": "BLOCK", "basis": rule.basis, "role": rule.role, "assignee": assignee,
                    "message": f"{label}: {assignee}" if assignee else f"{label}를 지정해야 승인 요청을 할 수 있습니다.",
                })
            else:
                evaluations.append({
                    "rule_code": rule.code, "title": rule.title, "status": "CHECK", "severity": "WARN",
                    "message": rule.message, "basis": rule.basis,
                })

    blocked = any(e["severity"] == "BLOCK" and e["status"] != "PASS" for e in evaluations)
    return {
        "blocked": blocked,
        "evaluations": evaluations,
        "conflicts": [],
        "notes": ["[mock] 같은 구역·시간대 다른 허가와의 충돌은 검사하지 않았습니다 (C 규칙 엔진에서 업무 기록 조회 필요)."],
    }


CHECK_RULES_PARAMETERS = {
    "type": "object",
    "properties": {
        "permit_types": {"type": "array", "items": {"type": "string"}},
        "measurements": {"type": "object"},
        "assignments": {"type": "object"},
        "location": {"type": ["string", "null"]},
        "period": {"type": "object"},
    },
    "required": ["permit_types"],
}
