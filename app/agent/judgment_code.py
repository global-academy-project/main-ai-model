"""④ 판단 (코드 방식): 위험요인·대책·점수는 템플릿, 근거(인용)는 위험요인마다 다시 검색해 코드가 고른다. LLM을 쓰지 않는다.

judgment_pick.py는 LLM이 근거 번호를 골랐다. 작은 모델은 관련 없는 조문을 고르거나(화재 위험요인에 제296조·제350조),
고르는 데만 100초 가까이 걸렸다. 이 모듈은 그 일을 코드로 바꾼다.

  1. 위험요인    rule_hazards() 그대로 (judgment.py). 위험요인·빈도·강도·대책·MSDS 인용은 바뀌지 않는다.
  2. 다시 검색   위험요인마다 질의를 만들어 기존 search_guides·search_accidents 도구를 부른다 (질의는 아래 SPECS).
                 처음 근거 수집은 "작업 전체"로 찾은 것이라 위험요인 하나하나에 맞는 조문·사례가 밀려 있을 수 있다.
  3. 고르기      다시 찾은 결과 + 처음 수집한 근거를 한 풀로 합쳐, 본문·제목의 키워드 점수로 줄을 세운다.
                 하이브리드 검색은 관련이 없어도 상위 k개를 돌려주므로(점수가 순위 기반), 점수가 기준(MIN_SCORE) 미만이면 버린다.
                 자리는 법령·지침 최소 2건 + 사고사례 최소 1건(있을 때)을 먼저 점수순으로 채우고, 나머지는 점수순으로 채운다 (최대 MAX_CITATIONS).
  4. 대체        고를 근거가 없거나 검색이 모두 실패하면 템플릿의 이전 키워드 인용을 그대로 둔다 (실행을 멈추지 않는다).

키워드는 judgment.py의 _TEMPLATES.keywords("용접", "환기" 같은 흔한 단어 포함)보다 엄격하게 따로 둔다 (SPECS의 strong/weak).
judgment.py는 고치지 않는다. verify()가 쓰는 템플릿 인용은 이전 그대로다.

judgment 출처는 "fallback"이다 (LLM을 쓰지 않았다는 뜻. 응답 스키마의 값은 "llm"|"fallback"뿐이라 바꾸지 않았다).
위험요인 자체의 출처(Hazard.source)는 "rule" 그대로다.

끄기: JUDGMENT_CODE = False 이면 work_register가 이 모듈을 부르지 않고 이전 방식(pick 또는 full)으로 돌아간다.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.exceptions import ToolError
from app.tools.tool_contract import ToolInvoker

from .evidence_collector.models import EvidenceItem, SearchToolOutput
from .evidence_collector.vocab import DEFAULT_CORPORA
from .judgment import Hazard, JudgeInput, kb_citation, rule_hazards

logger = logging.getLogger(__name__)

JUDGMENT_CODE = True  # True: 이 모듈 / False: 이전 방식 (pick 또는 full)
PER_HAZARD_K = 5  # 자료 묶음마다 위험요인 질의 한 번에 가져올 건수
MAX_CITATIONS = 3  # 위험요인 하나에 붙이는 근거 상한 (judgment.template_hazard와 같다)
MIN_PER_CORPUS = {"guide": 2, "accident": 1}  # 자리 배분: 법령·지침 최소 2건 + 사고사례 최소 1건 (있을 때). 사고사례 점수가 높아도 조문 자리를 뺏지 못한다
MIN_SCORE = 2  # 이 점수 미만의 근거는 버린다. strong 1개 = 2점, weak 1개 = 1점 (내가 정한 휴리스틱)
STRONG_WEIGHT = 2
WEAK_WEIGHT = 1


@dataclass(frozen=True)
class HazardSpec:
    """위험요인 템플릿 하나의 다시 검색 질의와 점수 키워드."""

    guide_query: str  # search_guides (법령·기술지침)
    accident_query: str  # search_accidents (사고사례)
    strong: tuple[str, ...]  # 이 위험요인에만 나오는 말 (1개로 MIN_SCORE 통과)
    weak: tuple[str, ...]  # 관련은 있지만 흔한 말 (여러 개 겹쳐야 통과)


SPECS: dict[str, HazardSpec] = {
    "hot_ignition": HazardSpec(
        "화기작업 불티 가연물 화재감시자 비산방지",
        "용접 용단 화기작업 화재 폭발 사고",
        ("불티", "화재감시자", "비산", "가연물", "인화성", "가연성"),
        ("화재", "폭발", "용접", "용단", "화기"),
    ),
    "hot_fume": HazardSpec(
        "용접 흄 유해가스 환기 국소배기 호흡보호구",
        "용접 흄 유해가스 중독 환기 사고",
        ("흄", "국소배기", "호흡보호", "방진마스크", "방독마스크"),
        ("용접", "용단", "환기", "마스크", "유해가스"),
    ),
    "grinder": HazardSpec(
        "그라인더 연삭기 방호덮개 연삭숫돌 덮개",
        "그라인더 연삭 절단 베임 사고",
        ("그라인더", "연삭기", "연삭숫돌", "연마석", "방호덮개"),
        ("절단", "베임", "파편", "덮개"),
    ),
    "pipe_open": HazardSpec(
        "배관 개방 퍼지 맹판 잔류 물질 잠금",
        "배관 플랜지 교체 잔류 가스 누출 폭발 사고",
        ("퍼지", "맹판", "잔류", "플랜지"),
        ("배관", "누출", "밸브", "차단", "잠금"),
    ),
    "confined": HazardSpec(
        "밀폐공간 산소농도 적정공기 환기 감시인 송기마스크",
        "밀폐공간 질식 산소결핍 사고",
        ("밀폐공간", "질식", "산소결핍", "적정공기", "송기마스크", "감시인"),
        ("산소", "환기", "유해가스", "맨홀"),
    ),
    "height": HazardSpec(
        "고소작업 추락 방지 안전대 작업발판 안전난간",
        "고소작업 추락 떨어짐 사고",
        ("추락", "안전대", "작업발판", "안전난간", "떨어짐"),
        ("높이", "고소", "개구부", "사다리"),
    ),
    "electric": HazardSpec(
        "정전 작업 감전 방지 검전 잠금",
        "감전 사고 전기 작업",
        ("감전", "검전", "정전", "충전부", "접지"),
        ("전기", "절연", "잠금"),
    ),
    "lifting": HazardSpec(
        "양중 인양 줄걸이 신호수 낙하 방지",
        "인양 크레인 낙하 깔림 사고",
        ("줄걸이", "인양", "신호수", "양중", "크레인"),
        ("낙하", "깔림", "하부"),
    ),
    "general": HazardSpec(
        "작업 통로 정리정돈 넘어짐 미끄러짐",
        "넘어짐 미끄러짐 사고",
        ("넘어짐", "미끄러", "정리정돈", "통로"),
        ("정리", "넘어"),
    ),
}


def hazard_queries(h: Hazard, inp: JudgeInput) -> dict[str, str]:
    """{자료 묶음("guide"|"accident"): 질의}. 템플릿을 모르는 위험요인(LLM이 만든 것)은 빈 dict."""
    spec = SPECS.get(h.template or "")
    if spec is None:
        return {}
    accident = spec.accident_query
    if h.template == "hot_ignition":
        # 주변 인화성 물질 이름을 사고사례 질의에 붙인다 (예: 용접 용단 화기작업 톨루엔 화재 폭발 사고)
        names = [m.name for m in inp.msds if m.found and m.name and any("인화성" in g for g in m.ghs_classes)]
        if names:
            accident = f"화기작업 {' '.join(names)} 화재 폭발 사고"
    return {"guide": spec.guide_query, "accident": accident}


def score(spec: HazardSpec, item: EvidenceItem) -> int:
    """본문·제목에 키워드가 몇 개 있나 (같은 말은 한 번만 센다)."""
    haystack = f"{item.source}\n{item.text}"
    return (
        STRONG_WEIGHT * sum(1 for kw in spec.strong if kw in haystack)
        + WEAK_WEIGHT * sum(1 for kw in spec.weak if kw in haystack)
    )


class _Searcher:
    """위험요인 질의를 검색 도구로 보낸다. 같은 (묶음, 질의)는 한 번만. ToolError는 세고 계속한다."""

    def __init__(self, tools: ToolInvoker, k: int) -> None:
        self._tools = tools
        self._k = k
        self._cache: dict[tuple[str, str], list[EvidenceItem]] = {}
        self.calls = 0
        self.failures = 0

    def search(self, corpus: str, query: str) -> list[EvidenceItem]:
        key = (corpus, query)
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        c = DEFAULT_CORPORA[corpus]
        items: list[EvidenceItem] = []
        try:
            raw = self._tools.call(c.tool, {**c.arguments, "query": query, "k": self._k})
            if isinstance(raw, BaseModel):
                raw = raw.model_dump()
            hits = SearchToolOutput.model_validate(raw).hits
            items = [
                EvidenceItem(id=h.id, doc_id=h.doc_id, source=h.source, text=h.text, score=h.score,
                             corpus=corpus, round=0, matched_by=["hazard"])
                for h in hits
            ]
        except ToolError as exc:
            self.failures += 1
            logger.warning("위험요인 근거 검색 실패: corpus=%s tool=%s code=%s", corpus, c.tool, exc.code, exc_info=exc)
        except ValidationError as exc:
            self.failures += 1
            logger.warning("위험요인 근거 검색 출력이 규격과 다릅니다: tool=%s", c.tool, exc_info=exc)
        self._cache[key] = items
        return items


def select_citations(spec: HazardSpec, pool: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """풀에서 점수가 기준 이상인 것을 골라 최대 MAX_CITATIONS개.

    먼저 묶음마다 최소 자리(MIN_PER_CORPUS: 법령·지침 2, 사고사례 1)를 점수순으로 채우고, 남는 자리는 묶음 구분 없이 점수순으로 채운다.
    사고사례 본문은 키워드를 많이 담고 있어 점수가 높게 나오는데, 그대로 자르면 핵심 조문(예: 화재감시자)이 밀려난다.
    """
    seen: set[str] = set()
    scored: list[tuple[int, int, EvidenceItem]] = []  # (점수, 풀 안 순서, 근거)
    for order, item in enumerate(pool):
        if item.id in seen:
            continue
        seen.add(item.id)
        if (s := score(spec, item)) >= MIN_SCORE:
            scored.append((s, order, item))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen: list[tuple[int, int, EvidenceItem]] = []
    for corpus, minimum in MIN_PER_CORPUS.items():
        chosen += [t for t in scored if t[2].corpus == corpus][:minimum]
    chosen = chosen[:MAX_CITATIONS]
    for t in scored:
        if len(chosen) >= MAX_CITATIONS:
            break
        if t not in chosen:
            chosen.append(t)
    chosen.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in chosen]


def judge_code(tools: ToolInvoker, inp: JudgeInput, *, k: int = PER_HAZARD_K) -> tuple[list[Hazard], str, int]:
    """(위험요인, 출처 "fallback", 버린 근거 수 0). judgment.judge()·judge_pick()과 같은 모양이다.

    LLM 호출이 없다. 검색 도구가 실패해도 예외를 올리지 않고 처음 수집한 근거와 템플릿 키워드 인용으로 계속한다.
    """
    hazards = rule_hazards(inp)
    searcher = _Searcher(tools, k)
    changed = 0
    for h in hazards:
        spec = SPECS.get(h.template or "")
        if spec is None:
            continue
        pool: list[EvidenceItem] = []
        for corpus, query in hazard_queries(h, inp).items():
            pool += searcher.search(corpus, query)
        pool += inp.pack.items  # 처음 수집한 근거도 같은 기준으로 겨룬다
        picked = select_citations(spec, pool)
        if not picked:
            logger.info("위험요인 '%s': 기준(%d점)을 넘는 근거가 없어 템플릿 키워드 인용을 둡니다.", h.template, MIN_SCORE)
            continue
        kept_msds = [c for c in h.citations if c.kind == "MSDS"]  # 템플릿이 붙인 물질 정보는 항상 남긴다
        h.citations = [kb_citation(i) for i in picked] + kept_msds
        h.needs_review = False
        changed += 1
    logger.info("근거 코드 선택: 위험요인 %d건 중 %d건 (검색 %d회, 실패 %d회)", len(hazards), changed, searcher.calls, searcher.failures)
    return hazards, "fallback", 0


__all__ = ["JUDGMENT_CODE", "judge_code"]
