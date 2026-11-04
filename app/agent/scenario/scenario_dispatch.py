"""시나리오 디스패처: 요청 하나를 분류하고, 시나리오에 맞는 모델과 처리기로 넘긴다.

    요청 ─ 시나리오를 직접 지정? ─ 예 → 그 시나리오로 (전용 화면·엔드포인트. 분류 모델을 부르지 않는다)
        │ 아니오
        ├ 진행 중인 대화(in_progress)? ─ 예 → 진행 상태 확인 (_confirm_in_progress)
        │                                   ├ 확인됨 → 같은 시나리오로 이어 감 (분류 모델을 부르지 않는다)
        │                                   └ 안 됨 → 이전 상태(slots)를 버리고 아래에서 새로 분류
        │ 아니오                                      (사유는 DispatchResult.continuation_rejected·로그)
        └→ `router` 모델이 시나리오 분류 (통합 채팅창, ScenarioRouter)
             ├ UNKNOWN → 되묻는 문장(ask)을 돌려준다 (처리기를 부르지 않는다)
             │          채팅 범위(chat_scenarios) 밖으로 분류되면 전용 화면을 안내한다 (decision.redirect)
             └→ 시나리오 보정 (_resolve): 등록 때 준 `resolve`가 최종 시나리오를 정한다 (세 경로 모두)
                  A·F는 현장 모드(payload["mode"])로 정한다: VERIFY → A, GUIDE → F (app/agent/site_mode.py)
                  보정 결과가 채팅 범위 밖이면 전용 화면 안내 (직접 지정한 요청은 범위를 보지 않는다)
                  ├ 처리기가 등록된 시나리오 → 그 시나리오의 모델(provider.get("A") 등)과 함께 처리기 호출
                  └ 처리기가 없는 시나리오 → handled=False, 담당자(owner)를 알려 준다

- 처리기(`ScenarioHandler`)는 엔진을 받아 쓰는 쪽이다. 어떤 모델인지 모르고 `llm`만 받는다.
  A(작업 등록)·F(안내)는 B가 등록하고, B·C·D·E는 각 담당자가 같은 방식으로 등록한다.
- LLMError·ToolError는 삼키지 않고 그대로 올린다 (HTTP 핸들러나 루프 러너가 처리).
- 처리기를 부를 때 모델 로딩이 일어나므로, 분류·되묻기 단계에서는 시나리오 모델을 만들지 않는다.
- 되묻기는 두 종류다. 둘은 다음 요청에 돌려보내는 값이 다르다.
    ① 시나리오 되묻기(분류가 안 됨): 앞선 말을 `payload["pending_text"]`로 돌려받아 이번 답과 이어 붙여 다시 분류한다.
       분류가 되면 처리기는 이어 붙인 문장 전체를 `request.message`로 받는다 (첫 발화가 사라지지 않게).
    ② 항목 되묻기(처리기가 WAITING_INPUT): `in_progress`와 `payload["slots"]`로 돌려받아 분류 모델 없이 같은 처리기로 이어 간다.
- 진행 상태 확인: `in_progress`는 클라이언트가 돌려보낸 값이라 그대로 믿지 않는다. 아래를 모두 만족해야 이어 간다.
    ⓐ 그 시나리오에 처리기가 있다 (처리기가 없으면 되묻기를 했을 리 없다)
    ⓑ 채팅 범위(chat_scenarios) 안이다
    ⓒ 시나리오 되묻기 값(pending_text)과 함께 오지 않았다 (두 되묻기는 동시에 일어나지 않는다)
    ⓓ 처리기를 등록할 때 준 `continuation_check`가 문제없다고 한다 (A·F는 slots·askedKeys 확인)
  하나라도 걸리면 이전 상태를 버리고 새 대화처럼 분류한다. 짧은 답("3")만 왔다면 대개 시나리오 되묻기가 된다.
  확인된 뒤에도 보정은 거친다: in_progress=A로 이어 가던 중 현장 모드가 GUIDE로 바뀌었으면 F로 이어 간다
  (A·F는 같은 에이전트·같은 slots라 상태가 그대로 맞는다).
- 분류는 `ScenarioClassifier`에 맡긴다 (참고 자료 검색 포함). 분류 API와 채팅 경로가 같은 문장을 같게 분류하도록.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any, Protocol

from app.llm.llm_base import LLMClient
from app.llm.llm_provider import LLMClientProvider

from .scenario_classifier import ScenarioClassifier
from .scenario_router import (
    LABELS,
    MAX_MESSAGE_CHARS,
    Scenario,
    ScenarioDecision,
    ScenarioRouter,
    build_ask,
    continuation,
    explicit,
)

logger = logging.getLogger(__name__)

PENDING_TEXT_KEY = "pending_text"  # ScenarioRequest.payload 키. 시나리오 되묻기 전에 받은 말
SLOTS_KEY = "slots"  # ScenarioRequest.payload 키. 항목 되묻기 상태 (직전 응답의 slots 그대로)
_MODE_PAYLOAD_KEY = "mode"  # 로그용. scenario_handlers.MODE_KEY와 같은 값 (거기서 가져오면 순환 임포트)


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

# 돌려받은 요청으로 이 시나리오를 이어 갈 수 있는지. 문제가 있으면 사유(로그·진단용 고정 문구), 없으면 None.
# 처리기 주인이 자기 상태 모양(payload["slots"])을 보고 정한다. 모델을 부르지 않는다.
ContinuationCheck = Callable[[ScenarioRequest], "str | None"]

# 분류된(또는 이어 가는·지정된) 시나리오를 최종 시나리오로 바꾼다. 같은 시나리오를 돌려주면 그대로.
# 서버가 확실히 아는 값(현장 모드 등)으로만 정한다. 모델을 부르지 않는다. UNKNOWN은 돌려줄 수 없다.
ScenarioResolver = Callable[[ScenarioRequest], Scenario]

@dataclass(frozen=True)
class DispatchResult:
    decision: ScenarioDecision
    handled: bool
    output: Any = None  # 처리기의 반환값
    ask: str | None = None  # 분류가 안 돼 되묻는 문장
    model: dict[str, str | None] | None = None  # 처리에 쓴 엔진·모델 (진단·steps 기록용)
    continuation_rejected: str | None = None  # in_progress를 받았지만 이어 가지 않은 사유 (진단용)
    resolved_from: Scenario | None = None  # 보정으로 시나리오가 바뀌었으면 바뀌기 전 시나리오 (진단용)

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
        self._continuation_checks: dict[Scenario, ContinuationCheck] = {}
        self._resolvers: dict[Scenario, ScenarioResolver] = {}
        for scenario, handler in (handlers or {}).items():
            self.register(scenario, handler)

    def register(
        self,
        scenario: Scenario,
        handler: ScenarioHandler,
        *,
        replace: bool = False,
        continuation_check: ContinuationCheck | None = None,
        resolve: ScenarioResolver | None = None,
    ) -> None:
        """continuation_check: 항목 되묻기를 이어 갈 때 상태를 확인하는 함수 (없으면 ⓐ~ⓒ만 본다).
        resolve: 이 시나리오로 정해졌을 때 최종 시나리오를 다시 정하는 함수 (없으면 그대로).
        replace=True면 두 함수도 함께 바뀐다 (주지 않은 함수는 지워진다)."""
        if scenario is Scenario.UNKNOWN:
            raise ValueError("UNKNOWN에는 처리기를 등록할 수 없습니다.")
        if scenario in self._handlers and not replace:
            raise ValueError(f"시나리오 {scenario.value}에는 이미 처리기가 있습니다. 바꾸려면 replace=True.")
        self._handlers[scenario] = handler
        for table, fn in ((self._continuation_checks, continuation_check), (self._resolvers, resolve)):
            if fn is None:
                table.pop(scenario, None)
            else:
                table[scenario] = fn

    @property
    def available_scenarios(self) -> tuple[Scenario, ...]:
        """처리기가 있고 채팅 범위 안인 시나리오 (안내 문구의 선택지용, 화면 표시 순서)."""
        return tuple(
            s for s in LABELS
            if s in self._handlers and (self._chat_scenarios is None or s in self._chat_scenarios)
        )

    def dispatch(self, request: ScenarioRequest) -> DispatchResult:
        request, rejected = self._confirm_in_progress(request)
        decision, text = self._classify(request)

        if (ask := decision.ask) is not None:
            self._log_decision(decision, request, None)
            return DispatchResult(decision, handled=False, ask=ask, continuation_rejected=rejected)

        if text != request.message:
            # 시나리오 되묻기 뒤에 분류된 경우: 처리기는 첫 발화까지 이어 붙인 문장을 받는다
            request = dc_replace(request, message=text)

        original = decision.scenario
        decision = self._resolve(decision, request)
        resolved_from = original if decision.scenario is not original else None
        self._log_decision(decision, request, resolved_from)  # 어떤 시나리오로 가는지 정해졌다
        if (ask := decision.ask) is not None:  # 보정 결과가 채팅 범위 밖
            return DispatchResult(
                decision, handled=False, ask=ask, continuation_rejected=rejected, resolved_from=resolved_from
            )

        handler = self._handlers.get(decision.scenario)
        if handler is None:
            logger.info("처리기 없는 시나리오: %s (담당 %s)", decision.scenario.value, decision.owner)
            return DispatchResult(
                decision, handled=False, continuation_rejected=rejected, resolved_from=resolved_from
            )

        key = decision.scenario.value
        llm = self._provider.get(key)  # 시나리오 모델은 여기서 처음 만들어진다 (보정 뒤의 시나리오로)
        binding = self._provider.binding_for(key)
        output = handler(request, llm)
        return DispatchResult(
            decision,
            handled=True,
            output=output,
            model={"engine": binding.engine, "model_id": binding.model_id},
            continuation_rejected=rejected,
            resolved_from=resolved_from,
        )

    def _log_decision(
        self, decision: ScenarioDecision, request: ScenarioRequest, resolved_from: Scenario | None
    ) -> None:
        """어떤 시나리오로 처리하게 됐는지 한 줄로 남긴다. 사용자 말은 남기지 않는다.

        scenario=UNKNOWN이면 처리기를 부르지 않고 되묻는다. redirect가 있으면 채팅 범위 밖이라 전용 화면을 안내한다.
        source: llm(모델 분류) | fallback(키워드 규칙) | continuation(이어 가기) | explicit(직접 지정) | empty(빈 입력)
        resolved_from: 현장 모드 보정으로 바뀌었으면 바뀌기 전 시나리오.
        """
        logger.info(
            "시나리오 결정: scenario=%s source=%s confidence=%.2f resolved_from=%s mode=%s handler=%s redirect=%s",
            decision.scenario.value,
            decision.source,
            decision.confidence,
            resolved_from.value if resolved_from else None,
            request.payload.get(_MODE_PAYLOAD_KEY),
            decision.scenario in self._handlers,
            decision.redirect.value if decision.redirect else None,
        )

    def _resolve(self, decision: ScenarioDecision, request: ScenarioRequest) -> ScenarioDecision:
        """등록된 resolve로 최종 시나리오를 정한다. 한 번만 적용한다 (보정 결과를 다시 보정하지 않는다)."""
        resolver = self._resolvers.get(decision.scenario)
        if resolver is None:
            return decision
        final = resolver(request)
        if not isinstance(final, Scenario) or final is Scenario.UNKNOWN:
            raise ValueError(f"시나리오 {decision.scenario.value}의 resolve가 잘못된 값을 돌려줬습니다: {final!r}")
        if final is decision.scenario:
            return decision
        logger.info("시나리오 보정: %s → %s (%s)", decision.scenario.value, final.value, decision.source)
        resolved = dc_replace(
            decision,
            scenario=final,
            reason=f"{decision.reason} (보정 {decision.scenario.value}→{final.value})"[:200],
        )
        if (
            decision.source != "explicit"
            and self._chat_scenarios is not None
            and final not in self._chat_scenarios
        ):
            return dc_replace(
                resolved,
                scenario=Scenario.UNKNOWN,
                candidate=None,
                redirect=final,
                ask=build_ask(redirect=final, allowed=self._chat_scenarios),
            )
        return resolved

    def _confirm_in_progress(self, request: ScenarioRequest) -> tuple[ScenarioRequest, str | None]:
        """in_progress가 가리키는 시나리오를 정말 이어 갈 수 있는지 확인한다.

        (요청, 거절 사유). 확인되면 요청 그대로와 None. 안 되면 in_progress와 이전 상태(slots)를 지운 요청과 사유
        → 새 대화처럼 분류된다. 사유에 사용자 말은 넣지 않는다.
        """
        scenario = request.in_progress
        if scenario is None:
            return request, None
        problem = self._continuation_problem(scenario, request)
        if problem is None:
            logger.info("진행 중인 시나리오 확인: %s → 이어 감", scenario.value)
            return request, None
        logger.warning("진행 중인 시나리오 %s를 이어 가지 않고 새로 분류: %s", scenario.value, problem)
        payload = {k: v for k, v in request.payload.items() if k != SLOTS_KEY}
        return dc_replace(request, in_progress=None, payload=payload), problem

    def _continuation_problem(self, scenario: Scenario, request: ScenarioRequest) -> str | None:
        if scenario not in self._handlers:
            return f"시나리오 {scenario.value}에는 처리기가 없습니다."
        if self._chat_scenarios is not None and scenario not in self._chat_scenarios:
            return f"시나리오 {scenario.value}는 채팅 범위 밖입니다."
        if (request.payload.get(PENDING_TEXT_KEY) or "").strip():
            return "시나리오 되묻기 값(pending_text)과 함께 왔습니다."
        if (check := self._continuation_checks.get(scenario)) is not None:
            return check(request)
        return None

    def _classify(self, request: ScenarioRequest) -> tuple[ScenarioDecision, str]:
        """(분류 결과, 처리기에 넘길 문장). 문장은 시나리오 되묻기 뒤에만 이어 붙인 것이고, 나머지는 원문 그대로."""
        if request.scenario is not None:
            return explicit(request.scenario), request.message  # 전용 화면: 분류 모델 없이
        if request.in_progress is not None:
            return continuation(request.in_progress), request.message  # 분류 모델 없이 이어 간다
        text = join_pending_text(request.payload.get(PENDING_TEXT_KEY), request.message)
        return self._classifier.classify(text).decision, text