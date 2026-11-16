"""규칙 판정 제목의 한글 보강 (TBM "…전에는 작업을 시작하지 않는다" 문장용, 11/16).

TBM 첫 교육 요점은 막힌 규칙의 제목(RuleEvaluationDto.title)을 이어 붙여 만든다. 제목은 C의 도구
(app/agent/rule_tools/check_rules.py)가 규칙 코드 → 한글 표(TITLES)로 붙이는데, 그 표에 없는 코드는 코드가
그대로 제목이 된다 (고소 3개·밀폐 3개가 11/02 시드에 추가됐지만 표에는 없다). 그래서 TBM에 영어 코드가 보였다.

C의 파일은 고치지 않는다. 표에 없어 제목이 한글이 아닌 항목만 여기서 보강한다.
  - 제목에 한글이 있으면 그대로 둔다 (C의 표가 우선. C가 코드를 표에 추가하면 이 표는 저절로 쓰이지 않는다).
  - 한글이 없으면 아래 표의 이름, 표에도 없으면 "규칙 판정" (영어 코드를 화면에 보이지 않는다). "규칙 판정"은 겹치면 한 번만 쓴다.
끄는 법: RULE_TITLE_FILL = False. 끄면 예전과 같다 (제목 그대로).

이름은 Spring 규칙 시드(R__rules_seed.sql)의 문구에 맞췄다. 기존 제목(화재감시자 지정, 소화기 비치)과 같은 짧은 명사형이다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.schemas.agent_work import RuleEvaluationDto

RULE_TITLE_FILL = True  # 한글이 아닌 규칙 제목을 한글로 바꾼다

GENERIC_TITLE = "규칙 판정"

# 규칙 코드 → 한글 이름 (C의 TITLES에 없는 것만). 근거 문구는 시드의 message
RULE_TITLES: dict[str, str] = {
    "CONFINED_WATCHER": "감시인 지정",  # 밀폐공간 외부에 배치할 감시인이 지정되지 않았습니다
    "CONFINED_VENTILATION": "작업 전 환기",  # 작업 시작 전 환기(적정공기 유지)가 확인되지 않았습니다
    "CONFINED_RESCUE_EQUIPMENT": "대피·구출용 기구 비치",  # 대피·구출용 기구 비치가 확인되지 않았습니다
    "HEIGHT_FALL_PROTECTION": "추락 방지 조치",  # 추락 방지 조치가 정해지지 않았습니다
    "HEIGHT_HELMET": "안전모 착용",  # 안전모 착용이 확인되지 않았습니다
    "HEIGHT_HARNESS": "안전대 착용",  # 안전대 착용이 확인되지 않았습니다
}

_HANGUL = re.compile("[가-힣]")


def display_title(rule_code: str, title: str) -> str:
    """화면에 쓸 규칙 제목. 한글이 있으면 그대로, 없으면 표의 이름 또는 "규칙 판정"."""
    if not RULE_TITLE_FILL or _HANGUL.search(title):
        return title
    return RULE_TITLES.get(rule_code) or GENERIC_TITLE


def blocking_titles(evaluations: Sequence[RuleEvaluationDto]) -> list[str]:
    """승인을 막는 판정 항목(BLOCK이고 PASS가 아님)의 제목. 순서는 판정 순서."""
    titles: list[str] = []
    for e in evaluations:
        if e.severity != "BLOCK" or e.status == "PASS":
            continue
        title = display_title(e.rule_code, e.title)
        if RULE_TITLE_FILL and title == GENERIC_TITLE and GENERIC_TITLE in titles:
            continue
        titles.append(title)
    return titles
