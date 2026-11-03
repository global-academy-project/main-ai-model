"""현장 모드(sites.mode) → 작업 등록 시나리오. 이 표 한 곳에서만 정한다.

계획서 7-5장: 안내 모드 GUIDE(50인 미만·안전관리자 없음) = 시나리오 F, 검증 모드 VERIFY(대형 현장) = 시나리오 A.
A·F 중 어느 쪽인지는 사용자의 말이 아니라 현장 설정으로 정한다 (분류기는 "작업 등록 계열인가"까지만 본다).

- Spring 실행 경로(`agent_work_service.run_work_register`)와 채팅 경로(`scenario_handlers`)가 같은 표를 쓴다.
- 값은 시나리오 코드 문자열("A"·"F")이다. `Scenario` enum을 임포트하지 않아 어느 계층에서 불러도 순환이 생기지 않는다.
- mode는 Spring이 `AgentRun.mode`로 넘긴 값이다. 요청자가 고르는 값이 아니다.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

SCENARIO_BY_MODE: Mapping[str, str] = MappingProxyType({"VERIFY": "A", "GUIDE": "F"})
MODE_BY_SCENARIO: Mapping[str, str] = MappingProxyType({s: m for m, s in SCENARIO_BY_MODE.items()})
SITE_MODES: tuple[str, ...] = tuple(SCENARIO_BY_MODE)


def scenario_key_for_mode(mode: object) -> str:
    """현장 모드에 맞는 시나리오 코드. 모르는 값이나 None이면 ValueError (요청 검증이 먼저 막아야 하는 값)."""
    try:
        return SCENARIO_BY_MODE[mode]  # type: ignore[index]
    except (KeyError, TypeError):
        raise ValueError(f"현장 모드는 {', '.join(SITE_MODES)} 중 하나여야 합니다: {mode!r}") from None
