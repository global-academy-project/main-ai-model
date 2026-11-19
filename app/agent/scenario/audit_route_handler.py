"""E(감사 대응) 처리기: 채팅에서 감사 보고서를 원하는 말을 감사 대응 화면으로 보낸다 (`AUDIT_ROUTE_ENABLED`).

이 처리기는 보고서를 만들지 않는다. 집계는 Spring DB에서 하고 기간은 채팅 문장에서 나오므로 일이 둘로 나뉜다.

    ① 이 처리기 (work-register 한 턴)   문장에서 기간을 읽는다 (`audit_period.py`)
         읽음   → runStatus=DONE, route={target: AUDIT_REPORT, dateFrom, dateTo, message}
         없음   → runStatus=WAITING_INPUT, ask(기간을 물음), slots.pendingText = 지금까지의 말
                  (시나리오 되묻기와 같은 방식: 다음 요청에서 이번 답과 이어 붙여 다시 분류한다. slots 모양은 그대로)
         여럿   → 위와 같고 "하나만 알려 달라"는 문구
    ② Spring·React (팀원)               route.target=AUDIT_REPORT이면 초안을 저장하지 않고, 그 기간으로 기존 감사 보고서
                                        만들기(집계 → /audit-report → /audit-report/compose, 이미 비동기)를 시작한다.
                                        채팅은 그 보고서 상태를 폴링하다가 끝나면 감사 대응 화면으로 가는 버튼을 보인다.

- 응답은 `WorkRegisterResult` 그대로다. 새 값은 선택 필드 `route` 하나뿐이고, 작업 등록 응답에서는 null이다.
- 모델을 쓰지 않는다 (디스패처가 시나리오 모델을 만들어 `llm`으로 넘기지만 받기만 한다).
- 화면 문구는 아래 상수 세 개뿐이다. 임시 문구이며 화면 문구는 팀이 정한다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

from app.llm.llm_base import LLMClient
from app.schemas.agent_work import (
    AskBack,
    Classification,
    MissingField,
    RouteInfo,
    WorkRegisterResult,
    WorkSlots,
)

from ..audit_period import read_audit_period
from ..work_date import seoul_today
from .scenario_dispatch import ScenarioHandler, ScenarioRequest

logger = logging.getLogger(__name__)

PERIOD_KEY = "audit_period"
ASK_MISSING = "어느 기간의 감사 대응 보고서를 만들까요? 예: 지난달, 이번 달, 10월, 2026-10-01~2026-10-31"
ASK_AMBIGUOUS = "기간이 둘 이상으로 읽혀요. 보고서에 담을 기간을 하나만 알려 주세요. 예: 지난달, 2026-10-01~2026-10-31"
ROUTE_MESSAGE = "{label} 기간의 감사 대응 보고서를 만들게요. 완료되면 감사 대응 화면에서 확인할 수 있어요."

# 처리기가 돌았을 때의 classification. 작업유형을 읽지 않았다 (high_risk=None은 "모름")
_NOT_CLASSIFIED = Classification(work_types=[], high_risk=None, required_permits=[], can_start_before_approval=False)


def make_audit_route_handler(*, today: Callable[[], date] = seoul_today) -> ScenarioHandler:
    """`today`는 기준일을 바꿔 시험하려는 자리다 (기본은 한국 시간 오늘)."""

    def handler(request: ScenarioRequest, llm: LLMClient) -> WorkRegisterResult:
        text = request.message  # 시나리오 되묻기 뒤에는 디스패처가 앞선 말과 이어 붙인 문장이다
        reading = read_audit_period(text, today())
        # 사용자 말은 남기지 않는다
        logger.info(
            "감사 대응 분기: run=%s site=%s problem=%s period=%s",
            request.payload.get("run_id"),
            request.payload.get("site_id"),
            reading.problem,
            reading.label,
        )
        if reading.ok:
            return WorkRegisterResult(
                run_status="DONE",
                slots=WorkSlots(),
                classification=_NOT_CLASSIFIED,
                route=RouteInfo(
                    target="AUDIT_REPORT",
                    date_from=reading.date_from.isoformat(),  # type: ignore[union-attr]
                    date_to=reading.date_to.isoformat(),  # type: ignore[union-attr]
                    message=ROUTE_MESSAGE.format(label=reading.label),
                ),
            )
        question = ASK_AMBIGUOUS if reading.problem == "AMBIGUOUS" else ASK_MISSING
        return WorkRegisterResult(
            run_status="WAITING_INPUT",
            slots=WorkSlots(pending_text=text[-2000:]),
            ask=AskBack(
                message=question,
                missing_fields=[MissingField(key=PERIOD_KEY, label="보고서 기간", question=question)],
            ),
            classification=_NOT_CLASSIFIED,
        )

    return handler
