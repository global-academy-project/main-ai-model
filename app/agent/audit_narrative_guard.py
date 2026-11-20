"""감사 보고서 항목 문장(LLM 후보)에 대한 규칙 가드 (B).

`AUDIT_NARRATIVE_LLM`을 켜면 D의 `generate_sentence_candidates`가 항목별 문장 후보를 만든다.
이 모듈은 후보를 받아 **코드 규칙으로만** 채택 여부를 정한다. 모델을 부르지 않고 D의 파일을 임포트하지 않는다.

가드가 막는 것 (이전에는 숫자 대조 하나뿐이었다)
  1. 평가어           위반·부적정 같은 말이 있으면 후보를 버린다 (구성기는 평가어를 만들지 않는다).
  2. 날짜             입력 기간·템플릿 문장에 없는 날짜(YYYY-MM-DD, "N월 N일")가 있으면 버린다.
  3. 핵심 값 누락     바꿔 쓸 템플릿 문장의 숫자가 후보에 하나라도 빠지면 버린다.
  4. 모양             문장 수·길이 제한.
  5. 교체 범위        기준 문장(첫 문장)·잘림 안내·근거(evidence) 문장·산식(÷) 문장은 템플릿을 그대로 둔다.
                      후보는 나머지 현황 문장만 대신한다.
  6. 호출 대상        상태가 REVIEW인 항목만 부른다 (확인 필요 없음·참고·확인 불가는 템플릿).
                      바꿀 문장이 없는 항목은 호출하지 않는다.
  7. 시간 예산        항목 사이에 합계 시간을 넘었으면 남은 항목은 템플릿으로 둔다.
숫자 대조(해당 항목 facts에 없는 숫자)는 이전과 같이 한다.

끄기: NARRATIVE_GUARD = False 이면 구성기가 이전 방식(상태 불문 호출·문장 전체 교체·숫자 대조만)으로 돌아간다.
평가어 목록(BANNED_WORDS)은 임시값이다. 안전 담당과 확정한다.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

NARRATIVE_GUARD = True  # False면 이전 방식
NARRATIVE_STATES: tuple[str, ...] = ("REVIEW",)  # LLM 문장을 받는 항목 상태
NARRATIVE_BUDGET_SECONDS = 30.0  # 항목 호출 사이에 확인하는 합계 시간
MAX_CANDIDATES = 4
MAX_SENTENCE_CHARS = 200
REQUIRE_ALL_NUMBERS = True  # 템플릿 현황 문장의 숫자를 후보가 모두 담아야 한다

# 임시값: 구성기는 위반 여부를 판정하지 않는다. 목록은 안전 담당과 확정한다.
BANNED_WORDS: tuple[str, ...] = (
    "위반", "부적정", "부적합", "불량", "미흡", "적발", "불이행", "불법",
    "과태료", "처벌", "과실", "책임", "심각", "문제점", "문제가",
)

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_KO_DATE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{1,2}\s*월")
_NUMBER = re.compile(r"(?<![0-9.])\d+(?:\.\d+)?")
_DATE_FOR_NUMBER = re.compile(r"\d{4}-\d{2}-\d{2}")
_TRUNCATED = re.compile(r"^전체 \d+건 중 \d+건만 표시함\.$")


def banned_words_in(text: str) -> list[str]:
    return [word for word in BANNED_WORDS if word in text]


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(_DATE_FOR_NUMBER.sub(" ", text))


def date_problems(text: str, allowed_dates: set[str]) -> list[str]:
    """입력에 없는 날짜 표현. ISO 날짜는 허용 목록에 있어야 하고, '10월 3일'·'2026년' 같은 표현은 쓰지 않는다."""
    problems = [d for d in _ISO_DATE.findall(text) if d not in allowed_dates]
    if _KO_DATE.search(text):
        problems.append("한글 날짜 표현")
    return problems


def is_protected(sentence: Any, *, first: bool) -> bool:
    """템플릿을 그대로 둘 문장: 기준 문장(첫 문장)·근거가 달린 문장·잘림 안내·산식(÷) 문장."""
    text = sentence.text
    return first or bool(sentence.evidence) or bool(_TRUNCATED.match(text)) or "÷" in text


def split_sentences(sentences: Sequence[Any]) -> tuple[list[Any], list[Any], list[Any]]:
    """(기준 문장, 후보가 대신할 문장, 그대로 둘 나머지)로 나눈다. 순서는 원래 순서를 지킨다."""
    head: list[Any] = []
    replaceable: list[Any] = []
    kept: list[Any] = []
    for index, sentence in enumerate(sentences):
        if index == 0:
            head.append(sentence)
        elif is_protected(sentence, first=False):
            kept.append(sentence)
        else:
            replaceable.append(sentence)
    return head, replaceable, kept


def check_candidates(
    candidates: Sequence[str],
    *,
    replaced_texts: Sequence[str],
    allowed_numbers: set[str],
    allowed_dates: set[str],
) -> tuple[str, str] | None:
    """문제가 있으면 (verification 코드, 사유), 없으면 None. 사유에 후보 원문은 싣지 않는다."""
    if not candidates or len(candidates) > MAX_CANDIDATES:
        return "NARRATIVE_REPLACED", "문장 수가 맞지 않아 템플릿을 유지함"
    if any(not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_SENTENCE_CHARS for text in candidates):
        return "NARRATIVE_REPLACED", "문장 길이가 맞지 않아 템플릿을 유지함"
    joined = " ".join(text.strip() for text in candidates)
    if banned_words_in(joined):
        return "NARRATIVE_REPLACED", "평가어가 있어 템플릿을 유지함"
    if date_problems(joined, allowed_dates):
        return "NARRATIVE_REPLACED", "입력에 없는 날짜가 있어 템플릿을 유지함"
    got = _numbers(joined)
    if any(number not in allowed_numbers for number in got):
        return "NARRATIVE_REPLACED", "문장 속 숫자가 항목 집계 값과 달라 템플릿을 유지함"
    if REQUIRE_ALL_NUMBERS:
        required = {number for text in replaced_texts for number in _numbers(text)}
        if required - set(got):
            return "NARRATIVE_REPLACED", "집계 값이 빠져 템플릿을 유지함"
    return None


def apply_guarded(
    built: Mapping[str, Any],
    generator: Callable[[Any], list[str]],
    *,
    allowed_numbers_of: Callable[[Any], set[str]],
    make_sentence: Callable[[str], Any],
    note: Callable[[str, str, str], None],
    date_from: str,
    date_to: str,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """항목마다 후보를 받아 가드를 통과한 것만 채택한다. 실패는 모두 템플릿 유지(보고서는 계속 만든다)."""
    started = clock()
    for entry in built.values():
        item = entry.item
        if item.state not in NARRATIVE_STATES:
            continue
        head, replaceable, kept = split_sentences(item.sentences)
        if not replaceable:
            continue
        if clock() - started > NARRATIVE_BUDGET_SECONDS:
            note("NARRATIVE_FALLBACK", "WARN", "문장 생성 시간 예산을 넘어 남은 항목은 템플릿 문장을 씀")
            return
        try:
            candidates = generator(item)
            if not isinstance(candidates, list):
                raise ValueError("후보 형식이 올바르지 않음")
        except Exception as exc:  # 모델 오류·형식 오류 모두 템플릿으로 계속한다
            note("NARRATIVE_FALLBACK", "WARN", f"{item.area}: 문장 생성 실패, 템플릿 문장을 사용함 ({type(exc).__name__})")
            continue
        template_dates = {d for s in item.sentences for d in _ISO_DATE.findall(s.text)}
        problem = check_candidates(
            candidates,
            replaced_texts=[s.text for s in replaceable],
            allowed_numbers=allowed_numbers_of(item),
            allowed_dates={date_from, date_to, *template_dates},
        )
        if problem is not None:
            code, reason = problem
            note(code, "WARN", f"{item.area}: {reason}")
            continue
        item.sentences = [*head, *(make_sentence(text.strip()) for text in candidates), *kept]
