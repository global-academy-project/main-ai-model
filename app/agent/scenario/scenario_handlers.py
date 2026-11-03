"""B가 맡은 시나리오 처리기: A(작업 등록), F(소규모 사업장 안내)를 `WorkRegisterAgent`에 연결한다.

- 처리기는 엔진을 받아 쓰는 쪽이다. 디스패처가 시나리오 모델(`provider.get("A")` 등)을 만들어 `llm`으로 넘긴다.
  이 모듈은 어떤 엔진·모델인지 모른다.
- 도구는 이름으로만 부른다 (`ToolInvoker`). mock인지 실제인지는 `tool_setup.py`가 정한다.
- A·F 중 어느 쪽인지는 현장 모드(`payload["mode"]`, Spring이 넘긴 sites.mode)로 정한다: VERIFY → A, GUIDE → F.
  분류기가 A를 골랐어도 GUIDE 현장이면 F로 간다 (`resolve_work_register_scenario`, 표는 `app/agent/site_mode.py`).
  그래서 A 처리기는 항상 VERIFY, F 처리기는 항상 GUIDE로 에이전트를 부른다.
- 에이전트 생성 인자는 `/work-register`의 `get_work_register_agent`와 맞춘다(`judge_max_tokens`). 설정은 가장자리
  (`scenario_deps`)에서 읽어 인자로 넘기고, 이 모듈은 `Settings`를 모른다.
- 처리기가 돌려주는 값은 `WorkRegisterResult` 그대로다. `runStatus == "WAITING_INPUT"`이면 항목 되묻기 중이다.
- 항목 되묻기를 이어 갈 때(in_progress=A·F) 디스패처가 `work_register_continuation_problem`으로 상태를 확인한다.
  직전 응답의 slots와 그 안의 askedKeys(직전에 되물은 항목)가 돌아와야 이어 간다.
- LLMError·ToolError는 삼키지 않고 그대로 올린다 (HTTP 핸들러나 루프 러너가 처리).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.llm.llm_base import LLMClient
from app.schemas.agent_work import WorkRegisterRequest, WorkRegisterResult

from ...tools.tool_contract import ToolInvoker
from ..site_mode import MODE_BY_SCENARIO, scenario_key_for_mode
from ..work_register import WorkRegisterAgent
from .scenario_dispatch import SLOTS_KEY, ScenarioDispatcher, ScenarioHandler, ScenarioRequest
from .scenario_router import Scenario

DEFAULT_MODE = "VERIFY"
MODE_KEY = "mode"  # ScenarioRequest.payload 키. 현장 모드 (Spring의 AgentRun.mode)


def resolve_work_register_scenario(request: ScenarioRequest) -> Scenario:
    """A·F로 정해진 요청의 최종 시나리오를 현장 모드로 정한다 (디스패처의 resolve).

    mode가 없거나 모르는 값이면 ValueError. 요청 검증(AgentChatRequest.mode 필수)이 먼저 막아야 하므로
    여기까지 왔다면 호출하는 쪽 버그다 (조용히 VERIFY로 두면 GUIDE 현장이 A로 처리된다).
    """
    return Scenario(scenario_key_for_mode(request.payload.get(MODE_KEY)))


def _asked_keys(slots: Any) -> Any:
    """slots에서 직전에 되물은 항목. `WorkSlots` 객체와 dict(snake·camel 둘 다)를 받는다."""
    if isinstance(slots, Mapping):
        return slots.get("asked_keys") or slots.get("askedKeys")
    return getattr(slots, "asked_keys", None)


def work_register_continuation_problem(request: ScenarioRequest) -> str | None:
    """A·F 항목 되묻기를 이어 갈 수 있는지 (디스패처의 continuation_check). 모델을 부르지 않는다.

    에이전트는 WAITING_INPUT을 낼 때 되물은 항목을 slots.askedKeys에 싣는다. 이것이 없으면
    짧은 답("3", "없음")을 어느 항목의 답으로 읽을지 모르므로 이어 가지 않는다.
    """
    slots = request.payload.get(SLOTS_KEY)
    if slots is None:
        return "slots가 없습니다 (직전 응답의 result.slots를 돌려보내야 합니다)."
    if not _asked_keys(slots):
        return "slots에 직전에 되물은 항목(askedKeys)이 없습니다."
    return None


def make_work_register_handler(
    tools: ToolInvoker, *, mode: str | None = None, judge_max_tokens: int | None = None
) -> ScenarioHandler:
    """`mode`를 고정하면(F는 "GUIDE") 요청의 mode를 무시하고, None이면 `payload["mode"]`를 따른다.

    `judge_max_tokens`가 None이면 에이전트 기본값을 쓴다.
    """
    agent_options = {} if judge_max_tokens is None else {"judge_max_tokens": judge_max_tokens}

    def handler(request: ScenarioRequest, llm: LLMClient) -> WorkRegisterResult:
        agent = WorkRegisterAgent(llm=llm, tools=tools, **agent_options)
        return agent.run(
            WorkRegisterRequest(
                message=request.message,
                slots=request.payload.get(SLOTS_KEY),
                mode=mode or request.payload.get(MODE_KEY) or DEFAULT_MODE,
            )
        )

    return handler


def register_work_register_handlers(
    dispatcher: ScenarioDispatcher, tools: ToolInvoker, *, judge_max_tokens: int | None = None
) -> None:
    """A·F 처리기를 디스패처에 등록한다 (이어 가기 상태 확인, 현장 모드 보정 포함)."""
    for scenario in (Scenario.WORK_REGISTER, Scenario.GUIDE):
        dispatcher.register(
            scenario,
            make_work_register_handler(
                tools, mode=MODE_BY_SCENARIO[scenario.value], judge_max_tokens=judge_max_tokens
            ),
            continuation_check=work_register_continuation_problem,
            resolve=resolve_work_register_scenario,
        )
