"""check_rules 실제 도구: Spring 규칙 엔진(/internal/v1/tools/check_rules)을 부른다.

입출력은 mock(mock_tools/mock_rules.py)과 같다 → work_register·drafting은 고칠 것이 없다.

입력 (mock과 같음 + 실행 값 3개)
  {"permit_types": ["HOT_WORK_PERMIT"],
   "measurements": {"O2": 20.9, "LEL": 0},
   "assignments": {"FIRE_WATCHER": "이름" | null, "ATTENDANT": ...},
   "form": {"extinguisher_ready": true, ...},          선택. 허가서 확인 항목 (화면에서 체크한 값)
   "site_id": 1,                                       필수. 실행 요청 값 (LLM이 정하지 않음)
   "run_id": 345,                                      선택. 판정 기록에 실행을 연결
   "permit_id": 12}                                    선택. 있으면 이 허가서에 판정을 저장 (허가 종류가 하나일 때만)
출력 (mock과 같음)
  {"blocked": bool,
   "evaluations": [{"rule_code", "title", "status": PASS|FAIL|MISSING, "severity": BLOCK|WARN, "message", "basis",
                    측정 항목이면 "item", "unit", "value", "criterion" / 배치 항목이면 "role", "assignee"}],
   "conflicts": [], "notes": [str], "evaluation_ids": [int]}

판정은 전부 Spring이 한다. 이 도구는 모양만 바꾼다:
  - Spring checks[] (적용 규칙 전부, 통과 포함) -> evaluations[]
  - blocked = Spring result가 BLOCK
  - 문구(message)·근거(basis)는 Spring 값 그대로. 통과 항목만 Spring이 문구를 주지 않아 여기서 짧게 만든다
  - 적용할 규칙이 없는 허가 종류(지금은 고소)는 판정 없이 notes에 남긴다 (Spring도 PASS·규칙 0개로 응답)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from app.exceptions import ToolError
from app.integrations import InternalApiClient

logger = logging.getLogger(__name__)

TOOL = "check_rules"

# 허가서 양식 -> Spring workType
WORK_TYPES: dict[str, str] = {
    "HOT_WORK_PERMIT": "HOT",
    "CONFINED_SPACE_PERMIT": "CONFINED",
    "HEIGHT_WORK_PERMIT": "HEIGHT",
}

# 허가 종류별 감시 역할 (form.watcher_name 으로 보낸다)
WATCHER_ROLES: dict[str, str] = {"HOT": "FIRE_WATCHER", "CONFINED": "ATTENDANT"}
ROLE_LABELS: dict[str, str] = {"FIRE_WATCHER": "화재감시자", "ATTENDANT": "감시인"}

# 측정 항목 단위 (Spring 규칙 시드와 같은 고정 단위). 허가서 초안이 받는 항목만
MEASUREMENT_UNITS: dict[str, str] = {"O2": "PCT", "CO2": "PCT", "LEL": "PCT_LEL", "H2S": "PPM", "CO": "PPM"}

# 규칙 코드 -> 화면 제목 (Spring rules 시드 R__rules_seed.sql 기준). 없으면 측정 항목 이름이나 규칙 코드
TITLES: dict[str, str] = {
    "CONFINED_O2_RANGE": "산소 농도",
    "CONFINED_CO2_MAX": "탄산가스 농도",
    "CONFINED_CO_MAX": "일산화탄소 농도",
    "CONFINED_H2S_MAX": "황화수소 농도",
    "CONFINED_LEL_MAX": "가연성 가스 농도",
    "HOT_LEL_MAX": "가연성 가스 농도",
    "HOT_FIRE_WATCHER": "화재감시자 지정",
    "HOT_EXTINGUISHER": "소화기 비치",
    "HOT_SPARK_PREVENTION": "불티 비산 방지",
    "HOT_FLAMMABLE_VAPOR_REMOVED": "인화성 증기 제거",
}
_ITEM_TITLES: dict[str, str] = {
    "O2": "산소 농도", "CO2": "탄산가스 농도", "LEL": "가연성 가스 농도", "H2S": "황화수소 농도", "CO": "일산화탄소 농도",
}

_MEASUREMENT = "measurement."
_WATCHER_TARGET = "form.watcher_name"


def make_check_rules(client: InternalApiClient) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """레지스트리에 등록할 handler를 만든다 (클라이언트는 기동 때 한 번 만든다)."""

    def check_rules(arguments: Mapping[str, Any]) -> dict[str, Any]:
        permit_types = arguments.get("permit_types", [])
        measurements = arguments.get("measurements") or {}
        assignments = arguments.get("assignments") or {}
        form = arguments.get("form") or {}
        if not isinstance(permit_types, list) or not all(isinstance(m, Mapping) for m in (measurements, assignments, form)):
            raise _input_invalid("check_rules 입력 형식이 맞지 않습니다.")
        if any(p not in WORK_TYPES for p in permit_types):
            raise _input_invalid("check_rules가 모르는 허가 종류가 있습니다.")

        site_id = _id(arguments.get("site_id"), required=True, label="site_id")
        run_id = _id(arguments.get("run_id"), required=False, label="run_id")
        permit_id = _id(arguments.get("permit_id"), required=False, label="permit_id")
        if permit_id is not None and len(permit_types) != 1:
            raise _input_invalid("permit_id는 허가 종류가 하나일 때만 넣을 수 있습니다.")

        evaluations: list[dict[str, Any]] = []
        notes: list[str] = []
        evaluation_ids: list[int] = []
        blocked = False
        for permit_type in permit_types:
            work_type = WORK_TYPES[permit_type]
            payload: dict[str, Any] = {
                "siteId": site_id,
                "workType": work_type,
                "input": {"measurement": dict(measurements), "form": _form(work_type, assignments, form)},
            }
            if run_id is not None:
                payload["agentRunId"] = run_id
            if permit_id is not None:
                payload["target"] = {"type": "PERMIT", "id": permit_id}

            data = client.call_tool(TOOL, payload, retry=permit_id is None)   # 저장하는 호출은 재시도하지 않음

            checks = data.get("checks")
            if not isinstance(checks, list):
                logger.warning("check_rules 응답에 checks가 없음: keys=%s", sorted(data))
                raise ToolError(code="TOOL_UNAVAILABLE", message="check_rules 서버 응답 모양이 맞지 않습니다.",
                                status_code=502, tool=TOOL)
            if data.get("rulesApplied") == 0:
                notes.append(f"{permit_type}: 아직 판정 규칙이 없어 판정하지 않았습니다.")
            if data.get("result") == "BLOCK":
                blocked = True
            if isinstance(data.get("evaluationId"), int):
                evaluation_ids.append(data["evaluationId"])
            evaluations += [_evaluation(c, work_type, measurements, assignments) for c in checks]

        return {"blocked": blocked, "evaluations": evaluations, "conflicts": [], "notes": notes,
                "evaluation_ids": evaluation_ids}

    return check_rules


def _form(work_type: str, assignments: Mapping[str, Any], form: Mapping[str, Any]) -> dict[str, Any]:
    """허가서 입력값. 감시 역할은 watcher_name 으로. 값이 없으면 넣지 않는다 (Spring이 MISSING으로 판정)."""
    out = {k: v for k, v in form.items() if v is not None}
    role = WATCHER_ROLES.get(work_type)
    watcher = assignments.get(role) if role else None
    if isinstance(watcher, str) and watcher.strip():
        out["watcher_name"] = watcher.strip()
    return out


def _evaluation(check: Mapping[str, Any], work_type: str, measurements: Mapping[str, Any],
                assignments: Mapping[str, Any]) -> dict[str, Any]:
    rule_code = str(check.get("ruleCode"))
    target = str(check.get("target") or "")
    status = check.get("status")
    expected = check.get("expected") or ""
    item = target[len(_MEASUREMENT):] if target.startswith(_MEASUREMENT) else None
    title = TITLES.get(rule_code) or (_ITEM_TITLES.get(item) if item else None) or rule_code

    e: dict[str, Any] = {
        "rule_code": rule_code,
        "title": title,
        "status": status,
        "severity": check.get("severity"),
        "message": check.get("message"),
        "basis": check.get("legalBasis") or "",
    }

    if item in MEASUREMENT_UNITS:
        value = measurements.get(item)
        e.update(item=item, unit=MEASUREMENT_UNITS[item], value=value, criterion=expected)
        if status == "PASS":
            e["message"] = f"{title} {value} — 기준({expected}) 충족."
    elif target == _WATCHER_TARGET and work_type in WATCHER_ROLES:
        role = WATCHER_ROLES[work_type]
        assignee = assignments.get(role)
        assignee = assignee.strip() if isinstance(assignee, str) and assignee.strip() else None
        e.update(role=role, assignee=assignee if status == "PASS" else None)
        if status == "PASS":
            e["message"] = f"{ROLE_LABELS[role]}: {assignee}"
    elif status == "PASS":
        e["message"] = f"{title}: 확인됨."

    if not e["message"]:
        e["message"] = f"{title}: 기준을 충족하지 않습니다."   # Spring이 문구를 안 준 경우 (규칙 문구가 비었을 때)
    return e


def _id(value: Any, *, required: bool, label: str) -> int | None:
    if value is None:
        if required:
            raise _input_invalid(f"check_rules에는 {label}가 필요합니다 (실행 요청 값으로 채움).")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _input_invalid(f"check_rules의 {label}는 1 이상의 정수여야 합니다.")
    return value


def _input_invalid(message: str) -> ToolError:
    return ToolError(code="TOOL_INPUT_INVALID", message=message, status_code=400, tool=TOOL)


CHECK_RULES_PARAMETERS = {
    "type": "object",
    "properties": {
        "permit_types": {"type": "array", "items": {"type": "string", "enum": list(WORK_TYPES)}},
        "measurements": {
            "type": "object",
            "description": "가스 측정값. 숫자로만. 단위 고정 (O2·CO2 %, LEL %LEL, 나머지 ppm). 모르는 값은 넣지 않는다",
            "properties": {k: {"type": "number"} for k in ("O2", "CO2", "LEL", "H2S", "CO", "NH3", "CL2")},
            "additionalProperties": False,
        },
        "assignments": {"type": "object", "description": "FIRE_WATCHER(화기)·ATTENDANT(밀폐) 이름"},
        "form": {"type": "object", "description": "허가서 확인 항목 (extinguisher_ready, spark_prevention, flammable_vapor_removed: true/false)"},
        "location": {"type": ["string", "null"]},
        "period": {"type": "object"},
    },
    "required": ["permit_types"],
}
