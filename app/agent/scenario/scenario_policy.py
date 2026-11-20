"""시나리오 판단 규칙: 채팅 입구가 요청을 어느 시나리오로 보낼지 **이 파일 한 곳에서** 정한다.

입구가 받는 시나리오는 A(작업 등록)·E(기록 조회·감사)·F(소규모 사업장 안내) 셋이다. 순서:

  1. 현장 모드가 GUIDE면 말을 보지 않고 곧바로 F (LLM·검색 없음).            → FORCED_BY_MODE
     (F가 채팅 범위 CHAT_SCENARIOS에 없으면 이 단계를 건너뛴다: GUIDE 현장도 아래 규칙으로 A·E를 가른다)
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
  AUDIT_QUERY_STRICT = False  기록 질문 분리 v1 규칙 그대로 ('자료·보고서'가 든 기록 질문도 E, E 낱말만 있는 말도 E).
  AUDIT_QUERY_SPLIT = False  기록 질문 분리 이전 규칙 그대로 (기록 조회성 문장도 E, '보고서'만 있어도 E).
  POLICY_ENABLED = False   이 파일이 아무것도 정하지 않는다 → 이전처럼 분류 LLM(router)이 모든 것을 정한다.
  POLICY_USE_LLM = True    규칙이 정하지 못한 말(되묻기 대상)을 되묻지 않고 분류 LLM에 맡긴다 (현장 모드 GUIDE→F는 그대로).

키워드는 임시값이다 (정답 세트와 안전 전문가 확인 전). 바꿀 곳은 아래 상수 세 개뿐이다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from .scenario_router import RECORD_QUERY_GUIDE, Scenario, ScenarioDecision, build_ask

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

# 감사 보고서를 말하는 문장은 '만들어/작성해'가 있어도 E다 (작업 등록의 서류는 위험성평가·허가서·점검표·TBM이고 보고서가 아니다).
# '감사합니다' 같은 인사말은 감사 기록 얘기가 아니므로 E 키워드를 찾기 전에 지운다 (작업 등록 문장 끝의 인사가 E가 되지 않게).
# '등록'·'신청'이 함께 있으면 이전 규칙대로 (정하지 않고 되묻는다).
# 끄기: AUDIT_PRIORITY = False → 이전 규칙 그대로 ('감사 보고서 만들어줘'는 되묻고, '감사합니다'도 E 키워드로 읽는다)
AUDIT_PRIORITY = True
AUDIT_WORDS: tuple[str, ...] = ("감사", "보고서")
REGISTER_WORDS: tuple[str, ...] = ("등록", "신청")
_THANKS_RE = re.compile(r"감사\s*(?:합니다|합니당|해요|했|드립|드려|히|하다|함)")

# 기록 질문 분리 (C 요청 2-3, T41): 화면은 작업 등록 탭(작업 서류·감사 대응 보고서)과 기록 질문 탭(기록 조회)으로 나뉜다.
# 그래서 감사 대응 보고서를 만들어 달라는 말만 E(보고서 분기)로 보내고, 기록을 묻는 말은 기록 질문 탭을 안내한다.
#   1) '감사' + 보고서 요청말 (등록·신청이 없을 때)                              → E (보고서 분기)
#   2) E 낱말 + 기록 질문말, 보고서 요청말·A의 강한 단서(등록·신청·N시)는 없음   → 기록 질문 안내 ('감사'가 있어도)
#   3) '감사' + '대응'                                                         → E (보고서 분기)
#   4) 그 밖에는 이전 규칙. 단 '보고서'는 E 낱말에서 뺀다 ('감사' 없이 '보고서'만 있으면 감사가 아니다)
# 2)에 'E 낱말이 있다'를 더한 것은 이전에 E로 가던 말만 안내로 돌리기 위해서다 (등록 문장에 '알려줘'가 있어도 A 그대로).
# 끄기: AUDIT_QUERY_SPLIT = False → 이 규칙 이전 그대로. (AUDIT_PRIORITY = False이면 함께 꺼진다)
AUDIT_QUERY_SPLIT = True
REPORT_WORDS: tuple[str, ...] = ("보고서", "자료", "만들", "작성", "생성")
RECORD_WORDS: tuple[str, ...] = (
    "실시율", "승인률", "조치율", "통계", "건수", "몇 건", "목록", "조회", "알려", "보여",
    "빠진", "누락", "안 한", "안한", "않은",  # 뒤 다섯 개는 B가 더한 임시 낱말 ('지난달 TBM을 안 한 작업은?' 같은 말)
)
E_STRONG_SPLIT: tuple[str, ...] = tuple(w for w in E_STRONG if w != "보고서")

# 분기 좁히기 v2 (AUDIT_QUERY_STRICT): 보고서 분기(E)는 '감사' + 보고서를 만들어 달라는 말일 때만 간다.
#   - 만들어 달라는 동사(만들·작성·생성)가 있어야 보고서 요청이다. 명사(보고서·자료)만 있으면 기록 질문말이 우선한다
#     ('감사 보고서 목록 보여줘', '지난달 실시율 통계 자료 알려줘' → 기록 질문 안내). 기록 질문말이 없으면 이전처럼 E.
#   - 기록 질문말에 이력·내역·기록을 더한다 (B가 더한 임시 낱말).
#   - E 낱말만 있고 기록 질문말도 보고서 요청도 없는 말('지난달 용접 작업')도 E로 보내지 않고 기록 질문 안내로 보낸다. (이전에는 E)
#     되묻지 않는 까닭: 되묻기 답은 앞선 말과 이어 붙여 다시 판단하므로 '지난달' + '작업 등록' 답이 다시 되묻기가 된다 (되풀이).
# 이전에 E로 가던 말만 바뀐다 (E → 안내). 등록·신청·N시·만들어·작성해와 만들·작성·생성이 든 말은 그대로다.
# 끄기: AUDIT_QUERY_STRICT = False → v1 규칙 그대로. (AUDIT_QUERY_SPLIT = False이면 함께 꺼진다)
AUDIT_QUERY_STRICT = True
REPORT_VERBS: tuple[str, ...] = ("만들", "작성", "생성")
REPORT_NOUNS: tuple[str, ...] = ("보고서", "자료")
RECORD_WORDS_STRICT: tuple[str, ...] = ("이력", "내역", "기록")


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


def _split_decision(
    audit_text: str, e: list[str], a_strong: list[str], allowed: tuple[Scenario, ...] | None
) -> ScenarioDecision | None:
    """기록 질문 분리 규칙 (AUDIT_QUERY_SPLIT). None이면 이전 규칙으로 간다.

    audit_text: 인사말을 지운 소문자 문장. e: 그 문장의 E 낱말('보고서' 제외). a_strong: A의 강한 단서(등록·신청·N시…).
    """
    audit = "감사" in audit_text
    strict = AUDIT_QUERY_STRICT
    report = _hits(audit_text, REPORT_VERBS if strict else REPORT_WORDS)  # strict: 만들어 달라는 동사만
    noun = _hits(audit_text, REPORT_NOUNS) if strict else []
    register = _hits(audit_text, REGISTER_WORDS)
    if audit and report and not register:
        return _scoped(
            ScenarioDecision(Scenario.AUDIT_QUERY, RULE_CONFIDENCE, "키워드 규칙: E 감사 보고서 요청", "fallback"), allowed
        )
    record = _hits(audit_text, RECORD_WORDS + (RECORD_WORDS_STRICT if strict else ()))
    # E가 채팅 범위 밖이면 안내 문구(감사 대응 보고서를 만들 수 있다)가 사실이 아니므로 이전처럼 범위 밖 안내로 간다
    in_chat = allowed is None or Scenario.AUDIT_QUERY in allowed
    if in_chat and e and record and not report and not a_strong:
        return ScenarioDecision(
            Scenario.UNKNOWN,
            RULE_CONFIDENCE,
            f"키워드 규칙: 기록 질문 {record[:3]} → 기록 질문 탭 안내"[:200],
            "fallback",
            redirect=Scenario.AUDIT_QUERY,
            ask=RECORD_QUERY_GUIDE,
        )
    if audit and noun and not register:  # 기록 질문말이 없는 '감사 보고서'·'감사 자료'
        return _scoped(
            ScenarioDecision(Scenario.AUDIT_QUERY, RULE_CONFIDENCE, "키워드 규칙: E 감사 보고서 요청", "fallback"), allowed
        )
    if audit and "대응" in audit_text and not register:
        return _scoped(
            ScenarioDecision(Scenario.AUDIT_QUERY, RULE_CONFIDENCE, "키워드 규칙: E 감사 대응", "fallback"), allowed
        )
    if strict and in_chat and e and not report and not a_strong:
        # 이전에는 E. 기록 질문말도 보고서 요청도 없고 E 낱말('지난달'·'감사' …)만 있는 말도 기록 질문 안내로 보낸다
        return ScenarioDecision(
            Scenario.UNKNOWN,
            RULE_CONFIDENCE,
            f"키워드 규칙: E 낱말 {e[:3]}만 있음 → 기록 질문 탭 안내"[:200],
            "fallback",
            redirect=Scenario.AUDIT_QUERY,
            ask=RECORD_QUERY_GUIDE,
        )
    return None


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
    if forced is not None and (allowed is None or forced in allowed):  # F가 채팅 범위 밖이면 강제하지 않는다
        decision = ScenarioDecision(forced, 1.0, f"현장 모드 {mode} → {forced.value} (말을 보지 않고 정함)", "explicit")
        return _scoped(decision, allowed)

    lowered = text.lower()
    audit_text = _THANKS_RE.sub(" ", lowered) if AUDIT_PRIORITY else lowered
    split = AUDIT_PRIORITY and AUDIT_QUERY_SPLIT
    e = _hits(audit_text, E_STRONG_SPLIT if split else E_STRONG)
    a_strong = _hits(lowered, A_STRONG)
    if (clock := _TIME_RE.search(lowered)) is not None:
        a_strong.append(clock.group(0))
    a_weak = _hits(lowered, A_WEAK)

    if split and (routed := _split_decision(audit_text, e, a_strong, allowed)) is not None:
        return routed

    if AUDIT_PRIORITY and not split and _hits(audit_text, AUDIT_WORDS) and not _hits(lowered, REGISTER_WORDS):
        return _scoped(
            ScenarioDecision(Scenario.AUDIT_QUERY, RULE_CONFIDENCE, "키워드 규칙: E 감사·보고서", "fallback"), allowed
        )

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
