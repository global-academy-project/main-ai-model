"""규칙 기반 문장 해석 도우미 (1주차: 되묻기 체크리스트 · classify_work가 쓴다).

계획서 8장 "되묻기는 필수 항목 체크리스트로 시작"에 맞춰 LLM 없이 키워드·정규식으로만 읽는다.
2주차에 LLM 판단(또는 태그 분류기 work_request 프로필)으로 바꿀 때 이 모듈을 대체한다.

사전(물질·장비·장소 단어)은 임시값이다. 현장 용어가 쌓이면 늘린다.
부정 표현("크레인은 안 씀")은 빼고, 조건부 표현("고착되면 그라인더로 절단")은 넣는다 (안전 쪽 판단).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# 물질: 표준 이름 → 다른 표기. 산소는 측정값과 헷갈리므로 넣지 않는다.
MATERIALS: dict[str, tuple[str, ...]] = {
    "톨루엔": ("톨루엔", "toluene"),
    "메탄올": ("메탄올", "메틸알코올", "methanol"),
    "에탄올": ("에탄올", "에틸알코올", "ethanol"),
    "아세톤": ("아세톤", "acetone"),
    "벤젠": ("벤젠", "benzene"),
    "자일렌": ("자일렌", "크실렌", "xylene"),
    "헥산": ("헥산", "노말헥산", "hexane"),
    "이소프로필알코올": ("이소프로필알코올", "IPA"),
    "시너": ("시너", "신나", "thinner"),
    "수소": ("수소",),
    "프로판": ("프로판", "LPG"),
    "메탄": ("메탄", "LNG", "도시가스"),
    "아세틸렌": ("아세틸렌",),
    "황화수소": ("황화수소", "H2S"),
    "암모니아": ("암모니아",),
    "염소": ("염소가스", "염소"),
    "질소": ("질소",),
    "휘발유": ("휘발유", "가솔린"),
    "경유": ("경유", "디젤"),
}

# 장비: 표준 이름 → 다른 표기
EQUIPMENT: dict[str, tuple[str, ...]] = {
    "그라인더": ("그라인더", "그라인다"),
    "용접기": ("용접기",),
    "토치": ("토치",),
    "가스절단기": ("가스절단기", "산소절단기"),
    "크레인": ("이동식 크레인", "크레인"),
    "호이스트": ("호이스트",),
    "체인블록": ("체인블록", "체인블럭"),
    "지게차": ("지게차",),
    "고소작업대": ("고소작업대", "스카이차", "렌탈"),
    "사다리": ("사다리",),
    "비계": ("이동식 비계", "비계"),
    "송풍기": ("송풍기", "환기팬"),
    "가스측정기": ("가스측정기", "가스 측정기"),
    "발전기": ("발전기",),
    "임팩트렌치": ("임팩트렌치", "임팩트 렌치"),
    "드릴": ("드릴",),
}

# 장소를 이루는 단어 (임시)
LOCATION_WORDS = frozenset({
    "반응기", "배관", "탱크", "펌프", "지붕", "옥상", "공장", "창고", "라인", "랙", "덕트", "보일러",
    "열교환기", "압축기", "컴프레서", "맨홀", "피트", "사일로", "구역", "플랫폼", "설비", "건물",
    "기계실", "전기실", "변전실", "저장조", "굴뚝", "컨베이어", "공정", "드럼", "노즐", "밸브", "타워", "칼럼",
})
POSITION_WORDS = frozenset({"하부", "상부", "내부", "외부", "위", "아래", "옆", "주변", "앞", "뒤", "측면", "입구", "상단", "하단", "안"})
_TAG_PATTERNS = (
    re.compile(r"^[A-Z]{1,4}-?\d{1,4}[A-Z]?$"),  # 설비 번호 (예: P-101)
    re.compile(r"^\d+호기$"),
    re.compile(r"^\d+(층|라인|공장)$"),
    re.compile(r"^[A-Z0-9]+동$"),
)
_PARTICLES = ("에서는", "에서", "에는", "으로", "에", "의", "은", "는", "이", "가", "을", "를", "로", "도")

NEGATION_AFTER = re.compile(
    r"^\s*(?:은|는|도|을|를|이|가)?\s*(?:"
    r"안\s*(?:씀|써|쓴|쓸|쓰|함|해|한)|사용\s*(?:안|하지|않|없)|미사용|없음|없이|필요\s*없|제외"
    r")"
)
SENTENCE_SPLIT = re.compile(r"[.!?\n]+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def is_negated(text: str, end: int) -> bool:
    """text[end:] 바로 뒤가 부정 표현이면 True ('크레인은 안 씀')."""
    return NEGATION_AFTER.match(text[end : end + 16]) is not None


def find_terms(text: str, vocab: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    """사전 단어를 찾는다. (언급 순서대로 쓰는 것, 부정된 것)."""
    found: list[tuple[int, str]] = []
    negated: list[tuple[int, str]] = []
    for canonical, aliases in vocab.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pos = _find_word(text, alias)
            if pos is None:
                continue
            target = negated if is_negated(text, pos + len(alias)) else found
            target.append((pos, canonical))
            break
    return [c for _, c in sorted(found)], [c for _, c in sorted(negated)]


def _find_word(text: str, word: str) -> int | None:
    """대소문자 무시 검색. 영문 약어(IPA, LPG)는 앞뒤가 영문이면 단어 일부로 보고 건너뛴다."""
    lowered, needle = text.lower(), word.lower()
    start = 0
    while (pos := lowered.find(needle, start)) != -1:
        if needle.isascii():
            before = lowered[pos - 1] if pos > 0 else " "
            after = lowered[pos + len(needle)] if pos + len(needle) < len(lowered) else " "
            if before.isascii() and before.isalnum() or after.isascii() and after.isalnum():
                start = pos + 1
                continue
        return pos
    return None


def material_of(token: str) -> str | None:
    for canonical, aliases in MATERIALS.items():
        if any(token.lower().startswith(a.lower()) for a in aliases):
            return canonical
    return None


def _is_tag(token: str) -> bool:
    return any(p.match(token) for p in _TAG_PATTERNS)


def _location_kind(token: str) -> str | None:
    if _is_tag(token):
        return "tag"
    if token in LOCATION_WORDS:
        return "loc"
    if token in POSITION_WORDS:
        return "pos"
    return None


def stem(token: str) -> tuple[str, str | None]:
    """조사를 떼고 장소 단어인지 본다. (원형, 종류) — 장소 단어가 아니면 종류 None."""
    if kind := _location_kind(token):
        return token, kind
    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) > len(particle):
            base = token[: -len(particle)]
            if kind := _location_kind(base):
                return base, kind
    return token, None


def extract_locations(text: str) -> tuple[str | None, dict[str, str]]:
    """(작업 장소, 물질 → 보관 위치).

    장소 단어가 이어지는 구간을 찾는다. 구간 바로 뒤에 물질이 오면('… 탱크에 톨루엔') 그 물질의 위치로,
    구간 앞이 '옆·인근·근처'면 주변 설명으로 보고 작업 장소로 쓰지 않는다. 남은 첫 구간이 작업 장소다.
    """
    location: str | None = None
    material_locations: dict[str, str] = {}
    for sentence in sentences(text):
        tokens = [t for t in re.split(r"[\s,;:]+", sentence) if t]
        i = 0
        while i < len(tokens):
            base, kind = stem(tokens[i])
            if kind not in ("tag", "loc"):
                i += 1
                continue
            run = [base]
            j = i + 1
            while base == tokens[i] and j < len(tokens):  # 첫 단어에 조사가 붙어 있으면 구간은 한 단어
                nxt, nkind = stem(tokens[j])
                if nkind is None:
                    break
                run.append(nxt)
                if nxt != tokens[j]:  # 조사가 붙어 있었으면 구간이 거기서 끝난다
                    j += 1
                    break
                j += 1
            phrase = " ".join(run)
            following = material_of(tokens[j]) if j < len(tokens) else None
            preceded_by_near = i > 0 and tokens[i - 1] in ("옆", "인근", "근처", "주변")
            if following:
                material_locations.setdefault(following, phrase)
            elif not preceded_by_near and location is None and any(stem(t)[1] in ("tag", "loc") for t in run):
                location = phrase
            i = j
    return location, material_locations


def unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
