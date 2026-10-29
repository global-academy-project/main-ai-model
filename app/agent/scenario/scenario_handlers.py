"""B가 맡은 시나리오 처리기: A(작업 등록), F(소규모 사업장 안내)를 `WorkRegisterAgent`에 연결한다.

- 처리기는 엔진을 받아 쓰는 쪽이다. 디스패처가 시나리오 모델(`provider.get("A")` 등)을 만들어 `llm`으로 넘긴다.
  이 모듈은 어떤 엔진·모델인지 모른다.
- 도구는 이름으로만 부른다 (`ToolInvoker`). mock인지 실제인지는 `tool_setup.py`가 정한다.
- F는 같은 에이전트를 안내 모드(GUIDE)로 부른다. A는 요청의 `mode`(사업장 설정)를 따르고, 없으면 VERIFY.
- 에이전트 생성 인자는 `/work-register`의 `get_work_register_agent`와 맞춘다(`judge_max_tokens`). 설정은 가장자리
  (`scenario_deps`)에서 읽어 인자로 넘기고, 이 모듈은 `Settings`를 모른다.
- 처리기가 돌려주는 값은 `WorkRegisterResult` 그대로다. `runStatus == "WAITING_INPUT"`이면 항목 되묻기 중이다.
- LLMError·ToolError는 삼키지 않고 그대로 올린다 (HTTP 핸들러나 루프 러너가 처리).
"""

from __future__ import annotations

from app.llm.llm_base import LLMClient
from app.schemas.agent_work import WorkRegisterRequest, WorkRegisterResult

from ...tools.tool_contract import ToolInvoker
from ..work_register import WorkRegisterAgent
from .scenario_dispatch import ScenarioDispatcher, ScenarioHandler, ScenarioRequest
from .scenario_router import Scenario

DEFAULT_MODE = "VERIFY"


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
                slots=request.payload.get("slots"),
                mode=mode or request.payload.get("mode") or DEFAULT_MODE,
            )
        )

    return handler


def register_work_register_handlers(
    dispatcher: ScenarioDispatcher, tools: ToolInvoker, *, judge_max_tokens: int | None = None
) -> None:
    """A·F 처리기를 디스패처에 등록한다."""
    dispatcher.register(
        Scenario.WORK_REGISTER, make_work_register_handler(tools, judge_max_tokens=judge_max_tokens)
    )
    dispatcher.register(
        Scenario.GUIDE, make_work_register_handler(tools, mode="GUIDE", judge_max_tokens=judge_max_tokens)
    )
