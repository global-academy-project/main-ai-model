"""시나리오 판단 규칙: 채팅 입구가 요청을 어느 시나리오로 보낼지 **이 파일 한 곳에서** 정한다.

입구가 받는 시나리오는 A(작업 등록)·E(기록 조회·감사)·F(소규모 사업장 안내) 셋이다. 순서:

  1. 현장 모드가 GUIDE면 말을 보지 않고 곧바로 F (LLM·검색 없음).            → FORCED_BY_MODE
  2. 그 밖에는 키워드 규칙으로 A와 E를 가른다 (LLM·검색 없음).                → E_STRONG / A_STRONG / A_WEAK
       E 키워드만 있다 → E        A 키워드만 있다 → A
       E와 A의 강한 키워드가 함께 있다, 또는 아무 키워드도 없다 → 정하지 않고 되묻는다 ("작업 등록인가요, 기록 조회인가요?")
  3. 되묻기에 대한 답은 디스패처가 앞선 말과 이어 붙여 이 규칙에 다시 넣는다 (기존 시나리오 되묻기 흐름 그대로).

이미 대화가 진행 중인 요청(in_progress)과 전용 화면이 직접 지정한 요청(scenario)은 분류를 거치지 않으므로 이 파일을 지나지 않는다.
채팅 범위(CHAT_SCENARIOS) 밖으로 정해지면 기존 분류기와 같이 전용 화면을 안내한다 (redirect).

source 값: 응답 스키마의 값(llm|fallback|continuation|explicit|empty)을 늘리지 않으려고 기존 값을 쓴다.
  현장 모드로 정한 F → "explicit" (호출한 쪽이 정한 값이다)   키워드로 정한 A·E, 되묻기 → "fallback" (키워드 규칙)
  어느 규칙이 골랐는지는 reason에 적힌다.

끄기:
  POLICY_ENABLED = False   이 파일이 아무것도 정하지 않는다 → 이전처럼 분류 LLM(router)이 모든 것을 정한다.
  POLICY_USE_LLM = True    규칙이 정하지 못한 말(되묻기 대상)을 되묻지 않고 분류 LLM에 맡긴다 (현장 모드 GUIDE→F는 그대로).

키워드는 임시값이다 (정답 세트와 안전 전문가 확인 전). 바꿀 곳은 아래 상수 세 개뿐이다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from .scenario_router import Scenario, ScenarioDecision, build_ask

POLICY_ENABLED = True
POLICY_USE_LLM = False

# 현장 모드가 정하는 시나리오. 이 모드면 말을 보지 않는다. (VERIFY는 A·E를 말로 가른다)
FORCED_BY_MODE: dict[str, Scenario] = {"GUIDE": Scenario.GUIDE}

# 과거 기록·감사·통계를 묻는 말. 있으면 E 쪽 (소문자로 비교)
E_STRONG: tuple[str, ...] = (
    "감사", "이력", "통계", "실시율", "조치율", "보고서", "내역", "조회", "기록",
    "지난달", "지난주", "지난번", "저번", "작년", "어제", "몇 건",
)
# 작업을 등록·신청하는 말. E 키워드와 함께 있으면 정하지 않고 되묻는다
A_STRONG: tuple[str, ...] = ("등록", "만들어", "작성해", "신청")
_TIME_RE = re.compile(r"\d{1,2}\s*시")  # "13시", "2시부터" 같은 작업 시각도 A의 강한 단서
# 작업 내용을 담은 말. E 키워드가 없으면 A
A_WEAK: tuple[str, ...] = (
    "작업", "화기", "밀폐", "고소", "용접", "절단", "그라인더", "교체", "정비", "보수", "설치", "투입",
    "허가", "위험성평가", "점검표", "tbm",
)

RULE_CONFIDENCE = 0.8  # 키워드 규칙으로 정한 결과의 확신도 (LLM 확신도와 같은 0~1 척도의 고정값)


def _hits(text: str, words: Iterable[str]) -> list[str]:
    return [w for w in words if w in text]


def _scoped(decision: ScenarioDecision, allowed: tuple[Scenario, ...] | None) -> ScenarioDecision:
    """채팅 범위 밖이면 UNKNOWN으로 바꿔 전용 화면을 안내한다 (ScenarioRouter._apply_scope와 같은 규칙)."""
    if allowed is None or decision.scenario in allowed or decision.scenario is Scenario.UNKNOWN:
        return decision
    return replace(
        decision,
        scenario=Scenario.UNKNOWN,
        candidate=None,
        redirect=decision.scenario,
        reason=f"채팅 범위 밖({decision.scenario.value}): {decision.reason}"[:200],
        ask=build_ask(redirect=decision.scenario, allowed=allowed),
    )


def _ask(reason: str, allowed: tuple[Scenario, ...] | None) -> ScenarioDecision:
    # 선택지에서 F는 뺀다: F는 현장 모드로 정해지므로 VERIFY 사용자에게 고르게 하지 않는다
    options = [s for s in (allowed or tuple(Scenario)) if s not in (Scenario.GUIDE, Scenario.UNKNOWN)]
    return ScenarioDecision(
        Scenario.UNKNOWN, 0.0, reason[:200], "fallback", ask=build_ask(None, allowed=options or None)
    )


def decide(text: str, *, mode: str | None = None, allowed: tuple[Scenario, ...] | None = None) -> ScenarioDecision | None:
    """규칙으로 정한 결과. None이면 이 파일이 정하지 않았으니 호출한 쪽이 이전 방식(분류 LLM)으로 간다.

    mode: 현장 모드(VERIFY|GUIDE) 또는 None(분류 API처럼 모드가 없는 호출). allowed: 채팅 범위(None이면 전부).
    모델도 검색도 부르지 않는다.
    """
    if not POLICY_ENABLED:
        return None
    text = (text or "").strip()
    if not text:
        return _scoped_unknown_empty(allowed)

    forced = FORCED_BY_MODE.get(mode or "")
    if forced is not None:
        decision = ScenarioDecision(forced, 1.0, f"현장 모드 {mode} → {forced.value} (말을 보지 않고 정함)", "explicit")
        return _scoped(decision, allowed)

    lowered = text.lower()
    e = _hits(lowered, E_STRONG)
    a_strong = _hits(lowered, A_STRONG)
    if (clock := _TIME_RE.search(lowered)) is not None:
        a_strong.append(clock.group(0))
    a_weak = _hits(lowered, A_WEAK)

    if e and not a_strong:
        return _scoped(ScenarioDecision(Scenario.AUDIT_QUERY, RULE_CONFIDENCE, f"키워드 규칙: E {e[:3]}", "fallback"), allowed)
    if a_strong and not e:
        return _scoped(
            ScenarioDecision(Scenario.WORK_REGISTER, RULE_CONFIDENCE, f"키워드 규칙: A {a_strong[:3]}", "fallback"), allowed
        )
    if a_weak and not e:
        return _scoped(
            ScenarioDecision(Scenario.WORK_REGISTER, RULE_CONFIDENCE, f"키워드 규칙: A {a_weak[:3]}", "fallback"), allowed
        )
    if POLICY_USE_LLM:
        return None
    reason = f"키워드로 A·E를 가리지 못함 (E {e[:2]}, A {(a_strong + a_weak)[:2]})" if e else "A·E 키워드가 없음"
    return _ask(reason, allowed)


def _scoped_unknown_empty(allowed: tuple[Scenario, ...] | None) -> ScenarioDecision:
    """빈 입력. ScenarioRouter와 같은 결과(source=empty)."""
    options = [s for s in (allowed or tuple(Scenario)) if s not in (Scenario.GUIDE, Scenario.UNKNOWN)]
    return ScenarioDecision(Scenario.UNKNOWN, 0.0, "빈 입력", "empty", ask=build_ask(None, allowed=options or None))


__all__ = ["decide", "POLICY_ENABLED", "POLICY_USE_LLM"]
