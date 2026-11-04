"""시나리오 모델 표의 키 이름 (계획서 9장 시나리오 A~F).

모델 표(`SCENARIO_ENGINES`·시나리오 모델 표), `scenario_llm(...)`에는
**이름**을 쓴다. 글자 코드("A"~"F")는 별칭으로 계속 받는다. 시나리오 분류기·디스패처는 내부에서 글자 코드
(`Scenario.WORK_REGISTER = "A"`)로 모델을 꺼내므로, 제공자가 글자를 이름으로 바꿔 같은 모델을 돌려준다.

| 글자 | 이름 | 계획서 9장 | 줄 주인 |
| --- | --- | --- | --- |
| A | work_register        | 짧은 요청 → 되묻기 → 아침 등록 항목(위험성평가·허가·점검표·TBM) 초안 | B |
| B | confined_space_gate  | 밀폐공간 작업 → 측정값이 기준 미달이면 승인 차단 (문장 → 값 추출, 7-4장) | C |
| C | near_miss_warning    | 아차사고 보고 → 5축 태깅 → 같은 구역 다음 고소작업에 경고 | A |
| D | daily_safety_log     | 마감 현황 대시보드 → TBM 기록 → 일일 안전일지 요약 | D |
| E | audit_query          | 감사 대응 기록 질의·보고서 (기록 질문 탭) | D |
| F | small_site_guide     | 안전관리자 없는 50인 미만 사업장 안내 모드 + 인쇄 | B |

`router`(시나리오 분류)와 `default`(줄이 없는 키)는 그대로다.
이 모듈은 다른 모듈을 임포트하지 않는다 (core·llm·agent 어디서든 불러도 된다).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

ROUTER_KEY = "router"
DEFAULT_KEY = "default"

# 글자 코드 → 이름. 이름을 바꿀 때는 이 표만 고친다
SCENARIO_NAMES: dict[str, str] = {
    "A": "work_register",
    "B": "confined_space_gate",
    "C": "near_miss_warning",
    "D": "daily_safety_log",
    "E": "audit_query",
    "F": "small_site_guide",
}
SCENARIO_CODES: dict[str, str] = {name: code for code, name in SCENARIO_NAMES.items()}

SCENARIO_KEY_NAMES: tuple[str, ...] = tuple(SCENARIO_NAMES.values())
# 모델 표에서 쓰는 이름 (router·default 포함). 표·describe()·로그는 이 이름으로 나온다
MODEL_KEYS: tuple[str, ...] = (ROUTER_KEY, DEFAULT_KEY, *SCENARIO_KEY_NAMES)

V = TypeVar("V")


def canonical_key(key: str) -> str:
    """이름은 그대로, 글자 코드는 이름으로. 모르는 키면 ValueError."""
    if key in MODEL_KEYS:
        return key
    if key in SCENARIO_NAMES:
        return SCENARIO_NAMES[key]
    allowed = ", ".join(MODEL_KEYS)
    raise ValueError(f"모르는 시나리오 키 {key!r} (이름: {allowed} / 글자 A~F도 받음)")


def is_model_key(key: str) -> bool:
    try:
        canonical_key(key)
    except ValueError:
        return False
    return True


def canonical_mapping(raw: Mapping[str, V], *, source: str = "SCENARIO_ENGINES") -> dict[str, V]:
    """키를 이름으로 바꾼 사본. 같은 시나리오를 글자와 이름으로 함께 적으면 ValueError (어느 쪽인지 모른다)."""
    out: dict[str, V] = {}
    origin: dict[str, str] = {}
    for key, value in raw.items():
        try:
            name = canonical_key(key)
        except ValueError as exc:
            raise ValueError(f"{source}: {exc}") from None
        if name in out:
            raise ValueError(f"{source}: '{origin[name]}'와 '{key}'는 같은 시나리오({name})입니다. 하나만 적으세요.")
        out[name] = value
        origin[name] = key
    return out


def describe_key(name: str) -> dict[str, Any]:
    """진단 표시용: 이름과 글자 코드."""
    return {"key": name, "code": SCENARIO_CODES.get(name)}
