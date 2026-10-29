"""시나리오 디스패처: 요청 하나를 분류하고, 시나리오에 맞는 모델과 처리기로 넘긴다.

    요청 ─ 시나리오를 직접 지정? ─ 예 → 그 시나리오로 (전용 화면·엔드포인트. 분류 모델을 부르지 않는다)
        │ 아니오
        ├ 진행 중인 대화? ─ 예 → 같은 시나리오로 이어 감 (분류 모델을 부르지 않는다)
        │ 아니오
        └→ `router` 모델이 시나리오 분류 (통합 채팅창, ScenarioRouter)
             ├ UNKNOWN → 되묻는 문장(ask)을 돌려준다 (처리기를 부르지 않는다)
             │          채팅 범위(chat_scenarios) 밖으로 분류되면 전용 화면을 안내한다 (decision.redirect)
             ├ 처리기가 등록된 시나리오 → 그 시나리오의 모델(provider.get("A") 등)과 함께 처리기 호출
             └ 처리기가 없는 시나리오 → handled=False, 담당자(owner)를 알려 준다

- 처리기(`ScenarioHandler`)는 엔진을 받아 쓰는 쪽이다. 어떤 모델인지 모르고 `llm`만 받는다.
  A(작업 등록)·F(안내)는 B가 등록하고, B·C·D·E는 각 담당자가 같은 방식으로 등록한다.
- LLMError·ToolError는 삼키지 않고 그대로 올린다 (HTTP 핸들러나 루프 러너가 처리).
- 처리기를 부를 때 모델 로딩이 일어나므로, 분류·되묻기 단계에서는 시나리오 모델을 만들지 않는다.
- 되묻기는 두 종류다. 둘은 다음 요청에 돌려보내는 값이 다르다.
    ① 시나리오 되묻기(분류가 안 됨): 앞선 말을 `payload["pending_text"]`로 돌려받아 이번 답과 이어 붙여 다시 분류한다.
       분류가 되면 처리기는 이어 붙인 문장 전체를 `request.message`로 받는다 (첫 발화가 사라지지 않게).
    ② 항목 되묻기(처리기가 WAITING_INPUT): `in_progress`로 돌려받아 분류 모델 없이 같은 처리기로 이어 간다.
- 분류는 `ScenarioClassifier`에 맡긴다 (참고 자료 검색 포함). 분류 API와 채팅 경로가 같은 문장을 같게 분류하도록.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Protocol

from app.llm.llm_base import LLMClient
from app.llm.llm_provider import LLMClientProvider

from .scenario_classifier import ScenarioClassifier
from .scenario_router import MAX_MESSAGE_CHARS, Scenario, ScenarioDecision, ScenarioRouter, continuation, explicit

logger = logging.getLogger(__name__)

PENDING_TEXT_KEY = "pending_text"  # ScenarioRequest.payload 키. 시나리오 되묻기 전에 받은 말


def join_pending_text(pending: str | None, message: str | None) -> str:
    """시나리오 되묻기 전의 말과 이번 답을 이어 붙인다. 분류기 입력 한도(2000자) 안에서 이번 답을 우선하고,
    넘치면 앞선 말의 앞쪽을 버린다 (가장 최근 말이 남는다)."""
    message = (message or "").strip()
    pending = (pending or "").strip()
    if not pending:
        return message[:MAX_MESSAGE_CHARS]
    room = MAX_MESSAGE_CHARS - len(message) - 1
    if room <= 0:
        return message[:MAX_MESSAGE_CHARS]
    return f"{pending[-room:]}\n{message}"

def parse_chat_scenarios(raw: str | Iterable[str] | None) -> tuple[Scenario, ...] | None:
    """`CHAT_SCENARIOS`("A,E,F") 같은 설정 값을 시나리오 목록으로. 없으면 None(전부). 잘못되면 ValueError."""
    if raw is None or raw == "":
        return None
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    result: list[Scenario] = []
    for item in items:
        code = str(item).strip().upper()
        try:
            scenario = Scenario(code)
        except ValueError:
            raise ValueError(f"CHAT_SCENARIOS에 모르는 시나리오가 있습니다: {code!r}") from None
        if scenario is Scenario.UNKNOWN:
            raise ValueError("CHAT_SCENARIOS에 UNKNOWN을 넣을 수 없습니다.")
        if scenario not in result:
            result.append(scenario)
    return tuple(result) or None

@dataclass(frozen=True)
class ScenarioRequest:
    message: str
    scenario: Scenario | None = None  # 전용 화면·엔드포인트가 직접 지정. 있으면 분류하지 않는다
    in_progress: Scenario | None = None  # 되묻기(WAITING_INPUT) 중인 시나리오. 이전 응답에서 받아 다시 보낸다
    payload: Mapping[str, Any] = field(default_factory=dict)  # 처리기가 쓰는 추가 입력 (slots, mode 등) + pending_text

    def __post_init__(self) -> None:
        if self.scenario is not None and self.in_progress is not None:
            raise ValueError("scenario와 in_progress는 함께 지정할 수 없습니다.")
        if self.scenario is Scenario.UNKNOWN:
            raise ValueError("scenario에 UNKNOWN을 지정할 수 없습니다.")

class ScenarioHandler(Protocol):
    def __call__(self, request: ScenarioRequest, llm: LLMClient) -> Any: ...

@dataclass(frozen=True)
class DispatchResult:
    decision: ScenarioDecision
    handled: bool
    output: Any = None  # 처리기의 반환값
    ask: str | None = None  # 분류가 안 돼 되묻는 문장
    model: dict[str, str | None] | None = None  # 처리에 쓴 엔진·모델 (진단·steps 기록용)

class _NoLLM(LLMClient):
    """설정 검증용. 호출되면 안 된다."""

    def chat(self, messages, **kwargs):  # pragma: no cover
        raise RuntimeError("검증용 엔진은 호출할 수 없습니다.")

_NO_LLM = _NoLLM()

class ScenarioDispatcher:
    def __init__(
        self,
        provider: LLMClientProvider,
        handlers: Mapping[Scenario, ScenarioHandler] | None = None,
        *,
        chat_scenarios: Iterable[Scenario] | None = None,
        classifier: ScenarioClassifier | None = None,
    ) -> None:
        """chat_scenarios: 통합 채팅창에서 분류해 줄 시나리오 (기본 전부). 나머지는 전용 화면이
        `ScenarioRequest(scenario=...)`로 직접 부른다. 잘못된 값은 ValueError (기동 실패).

        classifier: 분류를 맡길 `ScenarioClassifier` (참고 자료 검색 포함). 없으면 검색 없이 분류하는 것을 만든다.
        classifier를 직접 넘기면 채팅 범위는 그 classifier가 가진 값을 따른다."""
        self._provider = provider
        self._chat_scenarios = None if chat_scenarios is None else tuple(chat_scenarios)
        ScenarioRouter(_NO_LLM, allowed=self._chat_scenarios)  # 값 검증만 (기동 때 오류를 드러낸다)
        self._classifier = classifier or ScenarioClassifier(provider, None, chat_scenarios=self._chat_scenarios)
        self._handlers: dict[Scenario, ScenarioHandler] = {}
        for scenario, handler in (handlers or {}).items():
            self.register(scenario, handler)

    def register(self, scenario: Scenario, handler: ScenarioHandler, *, replace: bool = False) -> None:
        if scenario is Scenario.UNKNOWN:
            raise ValueError("UNKNOWN에는 처리기를 등록할 수 없습니다.")
        if scenario in self._handlers and not replace:
            raise ValueError(f"시나리오 {scenario.value}에는 이미 처리기가 있습니다. 바꾸려면 replace=True.")
        self._handlers[scenario] = handler

    def dispatch(self, request: ScenarioRequest) -> DispatchResult:
        decision, text = self._classify(request)

        if (ask := decision.ask) is not None:
            return DispatchResult(decision, handled=False, ask=ask)

        if text != request.message:
            # 시나리오 되묻기 뒤에 분류된 경우: 처리기는 첫 발화까지 이어 붙인 문장을 받는다
            request = dc_replace(request, message=text)

        handler = self._handlers.get(decision.scenario)
        if handler is None:
            logger.info("처리기 없는 시나리오: %s (담당 %s)", decision.scenario.value, decision.owner)
            return DispatchResult(decision, handled=False)

        key = decision.scenario.value
        llm = self._provider.get(key)  # 시나리오 모델은 여기서 처음 만들어진다
        binding = self._provider.binding_for(key)
        output = handler(request, llm)
        return DispatchResult(
            decision,
            handled=True,
            output=output,
            model={"engine": binding.engine, "model_id": binding.model_id},
        )

    def _classify(self, request: ScenarioRequest) -> tuple[ScenarioDecision, str]:
        """(분류 결과, 처리기에 넘길 문장). 문장은 시나리오 되묻기 뒤에만 이어 붙인 것이고, 나머지는 원문 그대로."""
        if request.scenario is not None:
            return explicit(request.scenario), request.message  # 전용 화면: 분류 모델 없이
        if request.in_progress is not None:
            return continuation(request.in_progress), request.message  # 분류 모델 없이 이어 간다
        text = join_pending_text(request.payload.get(PENDING_TEXT_KEY), request.message)
        return self._classifier.classify(text).decision, text
