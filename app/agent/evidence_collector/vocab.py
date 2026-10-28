"""어휘와 설정값: 작업유형, 자료 묶음(Corpus), 작업유형별 필수 근거(EvidenceRequirement).

DEFAULT_REQUIREMENTS의 항목·키워드는 임시값이며 안전 전문가·정답 세트로 확정한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# 태그 분류 체계(taxonomy.WORK_TYPES)와 같은 값. 태그 모듈에 의존하지 않으려고 따로 둔다.
WORK_TYPES: tuple[str, ...] = ("일반", "화기", "밀폐공간", "고소", "전기", "중량물")
_WORK_TYPE_SYNONYMS = {
    "화기작업": "화기",
    "용접": "화기",
    "밀폐": "밀폐공간",
    "밀폐공간작업": "밀폐공간",
    "고소작업": "고소",
    "전기작업": "전기",
    "중량물취급": "중량물",
    "양중": "중량물",
}

MAX_PLANNED_QUERIES = 8
MAX_QUERY_CHARS = 100
MAX_SUMMARY_CHARS = 120


def normalize_work_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = "".join(value.split())
    if key in WORK_TYPES:
        return key
    return _WORK_TYPE_SYNONYMS.get(key)


@dataclass(frozen=True)
class Corpus:
    """검색 대상 자료 묶음 하나 = 호출할 검색 도구 하나.

    arguments는 매 호출에 고정으로 실리는 추가 인자다 (나중에 태그 필터를 여기에 넣는다).
    query·k·min_score는 collector가 정하므로 arguments로 덮어쓸 수 없다.
    """

    description: str  # 계획 프롬프트에 그대로 들어간다
    tool: str  # 도구 레지스트리에 등록된 검색 도구 이름
    arguments: Mapping[str, Any] = field(default_factory=dict)


# 도구 이름은 A와 맞춘 값 (계획서 시나리오 A 도구 표).
DEFAULT_CORPORA: dict[str, Corpus] = {
    "accident": Corpus("재해조사보고서·아차사고 등 사고사례", "search_accidents"),
    "guide": Corpus("KOSHA GUIDE·위험성평가 안내서 등 작업 기준", "search_guides"),
}


@dataclass(frozen=True)
class EvidenceRequirement:
    """작업유형별로 꼭 있어야 하는 근거 하나.

    충족 기준(임시): corpus에서 찾은 근거 중 본문에 keywords 중 하나라도 들어 있는 것이 min_hits개 이상.
    query는 빠졌을 때 재질의의 기본 문장이다.
    """

    key: str
    description: str
    corpus: str
    query: str
    keywords: tuple[str, ...]
    min_hits: int = 1


_R = EvidenceRequirement
DEFAULT_REQUIREMENTS: dict[str, tuple[EvidenceRequirement, ...]] = {
    "화기": (
        _R("fire.accidents", "화기작업 중 화재·폭발 사고사례", "accident", "화기작업 화재 폭발 사고 사례", ("화재", "폭발", "불티")),
        _R("fire.gas_check", "가연성가스 측정 기준", "guide", "화기작업 가연성가스 농도 측정 기준", ("가연성가스", "가스 농도", "LEL")),
        _R("fire.watch", "화재감시자 배치 기준", "guide", "화기작업 화재감시자 배치", ("화재감시자",)),
    ),
    "밀폐공간": (
        _R("confined.accidents", "밀폐공간 질식·산소결핍 사고사례", "accident", "밀폐공간 질식 산소결핍 사고 사례", ("질식", "산소결핍")),
        _R("confined.air", "산소·유해가스 농도 기준", "guide", "밀폐공간 적정공기 산소 농도 기준", ("산소", "적정공기")),
        _R("confined.watch", "감시인 배치 기준", "guide", "밀폐공간 작업 감시인 배치", ("감시인",)),
    ),
    "고소": (
        _R("height.accidents", "추락 사고사례", "accident", "고소작업 추락 사고 사례", ("추락", "떨어짐")),
        _R("height.protection", "추락 방지 조치 기준", "guide", "고소작업 추락 방지 안전대 작업발판", ("안전대", "작업발판", "안전난간")),
    ),
    "전기": (
        _R("electric.accidents", "감전 사고사례", "accident", "전기작업 감전 사고 사례", ("감전",)),
        _R("electric.isolation", "정전·전원 차단 조치 기준", "guide", "전기작업 정전 전원 차단 잠금", ("정전", "차단", "잠금")),
    ),
    "중량물": (
        _R("lifting.accidents", "낙하·끼임 사고사례", "accident", "중량물 인양 낙하 끼임 사고 사례", ("낙하", "끼임", "깔림")),
        _R("lifting.rigging", "인양·줄걸이 작업 기준", "guide", "크레인 인양 줄걸이 작업 기준", ("줄걸이", "인양", "크레인")),
    ),
}
del _R
