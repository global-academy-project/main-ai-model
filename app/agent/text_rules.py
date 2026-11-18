"""규칙 기반 문장 해석 도우미 (1주차: 되묻기 체크리스트 · classify_work가 쓴다).

계획서 8장 "되묻기는 필수 항목 체크리스트로 시작"에 맞춰 LLM 없이 키워드·정규식으로만 읽는다.
2주차에 LLM 판단(또는 태그 분류기 work_request 프로필)으로 바꿀 때 이 모듈을 대체한다.

사전(물질·장비·장소 단어)은 임시값이다. 현장 용어가 쌓이면 늘린다.
부정 표현("크레인은 안 씀")은 빼고, 조건부 표현("고착되면 그라인더로 절단")은 넣는다 (안전 쪽 판단).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

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

# 현장 문장 측정(11/17)에서 사전 밖이라 놓친 말을 더한다 (T30). 임시값, 안전 담당 확인 전.
# 사다리차는 사다리가 아니다: 사전에 없어서 '사다리'로 잡혔다. 황산·염산·질산·초산은 흔한 약품이다.
# MSDS 조회어(msds_setup.LOOKUP_TERMS)는 PC에서 공단 실측 뒤에 더한다 (표에 없으면 이름 그대로 조회한다). False면 변경 전과 같다.
EXTENDED_VOCAB = True
if EXTENDED_VOCAB:
    EQUIPMENT["사다리차"] = ("사다리차",)
    MATERIALS["황산"] = ("황산",)
    MATERIALS["염산"] = ("염산",)
    MATERIALS["질산"] = ("질산",)
    MATERIALS["초산"] = ("초산", "아세트산")

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

# 부정은 같은 줄 안에서만 걸린다. [ \t]* 대신 \s*를 쓰면 줄바꿈을 넘어 다음 발화의 '없음'·'안 써요'가
# 앞 발화의 마지막 단어('용접')를 부정해 버린다 (발화를 \n으로 이어 붙여 분류하기 때문).
# 교체 표현('그라인더 말고 용접', '용접 대신 토치')도 앞 말을 쓰지 않는다는 뜻이다 (followup.py).
# '말고도'(그 밖에도)는 부정이 아니다. False면 변경 전과 같다.
REPLACE_NEGATION = True
_REPLACE_WORDS = r"|말고(?!도)|대신|아니라|아니고" if REPLACE_NEGATION else ""
# 정중형·완료형 부정('안 씁니다', '안 썼음', '쓰지 않음', '필요하지 않음', '투입하지 않음')도 부정이다 (T28). 사전 단어 바로 뒤에서만 본다.
# 동사는 쓰다·사용·투입·필요·동원·가동으로 한정한다 ('~하지 않'을 통째로 넓히면 '점검하지 않으면' 같은 말이 장비를 뺀다). False면 변경 전과 같다.
POLITE_NEGATION = True
_POLITE_WORDS = r"|안\s*(?:씁|썼|합|했)|(?:쓰|사용하|투입하|필요하|동원하|가동하)지\s*않|필요치\s*않" if POLITE_NEGATION else ""
# '사용 금지'·'쓸 일 없어서'·'쓸 계획이 없음'·'쓸 필요가 없음'도 쓰지 않는다는 뜻이다 (T30). 사전 단어 바로 뒤에서만 본다. False면 변경 전과 같다.
# '사용 금지 해제·표지·표시'는 금지가 풀렸거나 표지 이야기라 부정으로 보지 않는다 (작업 분류가 화기를 놓치지 않게).
PROHIBIT_NEGATION = True
_PROHIBIT_WORDS = r"|사용\s*금지(?!\s*(?:해제|해지|풀|표지|표시|푯말|스티커))|쓸\s*(?:일|계획|필요)\s*(?:이|가)?\s*없" if PROHIBIT_NEGATION else ""
NEGATION_AFTER = re.compile(
    r"^[ \t]*(?:은|는|도|을|를|이|가)?[ \t]*(?:"
    r"안\s*(?:씀|써|쓴|쓸|쓰|함|해|한)|사용\s*(?:안|하지|않|없)|미사용|없음|없이|필요\s*없|제외" + _REPLACE_WORDS + _POLITE_WORDS + _PROHIBIT_WORDS +
    r")"
)
SENTENCE_SPLIT = re.compile(r"[.!?\n]+")

# 긴 이름 안의 짧은 이름은 잡지 않는다 (T25). '메탄올'에서 '메탄'(다른 물질), '황화수소'에서 '수소'가 따로 잡혀
# 주변 물질·MSDS 조회·허가서 표에 잘못된 물질이 들어갔다. False면 변경 전과 같다 (부분 문자열이면 무엇이든 잡는다).
LONGEST_TERM_FIRST = True
# 사전에는 없지만 사전 단어를 품은 다른 물질 이름 (임시값, 안전 담당 확인 전). 이 이름 안에서는 사전 단어를 잡지 않는다.
# 과산화수소(수소 아님), 이·일·아산화질소(질소 아님), 염소산·차아염소산·과염소산(염소 아님), 이산화염소(염소 아님),
# 프로판올·이소프로판올(프로판 아님). 사전 안의 긴 이름(메탄올, 황화수소 등)은 따로 적지 않아도 같은 방식으로 걸러진다.
CONTAINING_NAMES = ("과산화수소", "산화질소", "염소산", "산화염소", "프로판올")
if EXTENDED_VOCAB:  # 황산·질산·초산·염산을 품은 다른 물질 (초산에틸은 에틸아세테이트, 아황산가스는 이산화황, 황산구리·질산칼륨은 염). 임시값
    CONTAINING_NAMES += (
        "초산에틸", "초산비닐", "아질산", "질산염", "질산암모늄", "질산칼륨", "질산나트륨", "아황산", "황산염", "황산화물", "과황산",
        "황산구리", "황산나트륨", "황산암모늄", "황산알루미늄", "황산칼슘", "황산마그네슘", "염산염",
    )


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def is_negated(text: str, end: int) -> bool:
    """text[end:] 바로 뒤가 부정 표현이면 True ('크레인은 안 씀')."""
    return NEGATION_AFTER.match(text[end : end + 16]) is not None


def find_terms(text: str, vocab: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    """사전 단어를 찾는다. (언급 순서대로 쓰는 것, 부정된 것)."""
    found: list[tuple[int, str]] = []
    negated: list[tuple[int, str]] = []
    covers = _covering_spans(text, vocab) if LONGEST_TERM_FIRST else ()  # 더 긴 이름이 차지한 구간 (T25)
    for canonical, aliases in vocab.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pos = _find_word(text, alias, covers)
            if pos is None:
                continue
            target = negated if is_negated(text, pos + len(alias)) else found
            target.append((pos, canonical))
            break
    return [c for _, c in sorted(found)], [c for _, c in sorted(negated)]


def _find_word(text: str, word: str, covers: Sequence[tuple[int, int]] = (), start: int = 0) -> int | None:
    """대소문자 무시 검색. 영문 약어(IPA, LPG)는 앞뒤가 영문이면 단어 일부로 보고 건너뛴다.

    covers: 더 긴 이름이 차지한 구간 (T25). 그 안에 통째로 든 자리는 건너뛴다. 비어 있으면 변경 전과 같다.
    start: 이 위치부터 찾는다.
    """
    lowered, needle = text.lower(), word.lower()
    while (pos := lowered.find(needle, start)) != -1:
        if needle.isascii():
            before = lowered[pos - 1] if pos > 0 else " "
            after = lowered[pos + len(needle)] if pos + len(needle) < len(lowered) else " "
            if before.isascii() and before.isalnum() or after.isascii() and after.isalnum():
                start = pos + 1
                continue
        if any(s <= pos and pos + len(needle) <= e and e - s > len(needle) for s, e in covers):
            start = pos + 1  # 더 긴 이름 안의 일부 (예: 메탄올 안의 메탄)
            continue
        return pos
    return None


def _spans(text: str, word: str) -> list[tuple[int, int]]:
    """word가 text에 나오는 모든 자리 (시작, 끝). 영문 약어 경계 규칙은 _find_word와 같다."""
    spans: list[tuple[int, int]] = []
    start = 0
    while (pos := _find_word(text, word, start=start)) is not None:
        spans.append((pos, pos + len(word)))
        start = pos + 1
    return spans


def _covering_spans(text: str, vocab: dict[str, tuple[str, ...]]) -> list[tuple[int, int]]:
    """사전의 모든 표기와 CONTAINING_NAMES가 text에서 차지한 구간. 이보다 짧은 단어는 그 안에서 잡지 않는다."""
    words = [alias for aliases in vocab.values() for alias in aliases] + list(CONTAINING_NAMES)
    return [span for word in words for span in _spans(text, word)]


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
