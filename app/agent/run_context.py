"""실행 요청에서 코드가 채우는 값.

Spring이 정해 넘긴 runId·siteId·permitId를 도구 인자에 넣는다. LLM 인자 스키마에는 넣지 않는다
(잘못된 현장 번호로 판정·저장되는 것을 막기 위해).
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: int | None = None
    site_id: int | None = None
    permit_id: int | None = None

    def check_rules_arguments(self, permit_types: Sequence[str]) -> dict[str, int]:
        """check_rules 인자에 덧붙일 값.

        - 값이 없는 키는 넣지 않는다. site_id가 없으면 실제 도구가 TOOL_INPUT_INVALID를 내고,
          에이전트는 판정 실패로 기록해 승인 요청을 막는다 (계획서 7-2, 안전하게 실패).
        - permit_id는 이번 작업의 허가 종류가 하나일 때만 넣는다. 넣으면 Spring이 판정을 저장하고
          재시도하지 않는다. 허가가 여럿이면 어느 허가서의 판정인지 정할 수 없어 미리 확인만 한다.

        permit_types: 이번 작업에서 필요한 허가 전체 (호출 하나의 허가가 아니라 분류 결과 전체).
        """
        args: dict[str, int] = {}
        if self.site_id is not None:
            args["site_id"] = self.site_id
        if self.run_id is not None:
            args["run_id"] = self.run_id
        if self.permit_id is not None and len(set(permit_types)) == 1:
            args["permit_id"] = self.permit_id
        return args