"""근거 계획 "줄이기"(brief) + 계획서 9장 순서(계획 → 물질 조회 → 근거 검색).

계획서 9장 시나리오 A 1단계는 LLM이 **작업·공구·물질**을 뽑는 것이다. 이 모듈은 계획 단계의 LLM을 그 범위로 줄인다.

    LLM   summary(작업 한 줄), equipment(공구), materials(물질)만 낸다. 짧아서 CPU에서도 빠르다.
          후보일 뿐이다: 공구·물질은 입력 원문에 실제로 나온 말만 채택하고, 부정된 것('크레인은 안 씀')은 뺀다.
    코드  작업 유형은 classify_work 결과 그대로. 검색 질의는 작업 유형별 필수 근거(DEFAULT_REQUIREMENTS)와
          MSDS 위험 분류로 조립한다 (계획서 4·5단계: "화기 + 인화성 + 화재·폭발", 화기작업 기준 확인).

순서(MSDS_FIRST): 계획 → lookup_msds → search_accidents·search_guides.
    classify_work는 되묻기 판단에 필요해서 계획보다 먼저 돈다(되묻기 중인 실행은 계획에 오지 않는다).

끄기: PLAN_MODE = "full" 이면 이전 방식(LLM이 검색 질의까지 만든다), MSDS_FIRST = False 이면 이전 순서
(근거 검색 → 물질 조회). 둘 다 끄면 변경 전과 같은 결과다 (tests/test_plan_brief.py).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence

from app.exceptions import ToolError
from app.llm.llm_base import LLMClient
from app.schemas.agent_work import MsdsDto
from app.tools.tool_contract import ToolInvoker

from .evidence_collector import EvidenceCollector, EvidencePack, SearchQuery, WorkContext
from .evidence_collector import sufficiency
from .evidence_collector.planner import Plan, Planner, build_planner_input
from .evidence_collector.searcher import SearchState
from .evidence_collector.textutil import extract_json_object, merge, one_line, str_list
from .evidence_collector.vocab import (
    DEFAULT_CORPORA,
    MAX_PLANNED_QUERIES,
    MAX_QUERY_CHARS,
    Corpus,
    EvidenceRequirement,
)
from .text_rules import EQUIPMENT, LOCATION_WORDS, MATERIALS, POSITION_WORDS, is_negated

logger = logging.getLogger(__name__)

# ── 켜고 끄는 값 (바꾼 뒤 서버를 다시 띄운다) ──
PLAN_MODE = "brief"  # "brief": LLM은 작업·공구·물질만, 질의는 코드. "full": 이전 방식
MSDS_FIRST = True  # True: 계획 → 물질 조회 → 근거 검색 (계획서 9장). False: 이전 순서 (근거 검색 → 물질 조회)

# ── 조정 값 ──
BRIEF_MAX_TOKENS = 160  # 계획 LLM 응답 최대 길이 (JSON 3키)
BRIEF_SUMMARY_CHARS = 60  # 작업 요약 최대 글자
MAX_BRIEF_ITEMS = 6  # 공구·물질 후보를 각각 이만큼까지만 받는다
MIN_ITEM_CHARS, MAX_ITEM_CHARS = 2, 20
MAX_HAZARD_TERMS = 2  # 검색 질의에 붙이는 MSDS 위험 분류 수
# 사고사례 질의에 MSDS 위험 분류를 붙이는 작업 유형 (고소·전기·중량물 질의에는 물질 분류가 맞지 않는다)
HAZARD_TERM_WORK_TYPES = ("화기", "밀폐공간")
# 질의에 붙일 만한 GHS 분류 (이 글자가 들어 있는 것만. 임시값, 실제 MSDS 분류 표기를 보고 맞춘다)
HAZARD_TERM_MARKERS = ("인화성", "급성 독성", "고압가스", "부식성", "발암성", "산화성")

_EXAMPLE_INPUT = "작업 내용: 내일 3호기 펌프 베어링 교체. 축이 고착돼 있으면 토치로 가열해서 뺄 거야. 옆에 메탄올 드럼 있음. 2명, 호이스트는 안 씀."
_EXAMPLE_OUTPUT = {"summary": "3호기 펌프 베어링 교체, 고착 시 토치 가열", "equipment": ["토치"], "materials": ["메탄올"]}
_RETRY_SUFFIX = "\n\n(이전 응답은 JSON으로 읽을 수 없었습니다. 위 형식의 JSON 객체 하나만 출력하세요.)"


def build_brief_prompt() -> str:
    return f"""당신은 산업현장 안전관리 에이전트의 작업 추출 담당입니다.
작업 내용을 읽고 JSON 객체 하나만 출력하세요. 설명은 쓰지 마세요.

[항목]
- summary: 작업이 무엇인지 한 줄 ({BRIEF_SUMMARY_CHARS}자 이내). 작업 내용에 있는 말로만 씁니다.
- equipment: 작업 내용에 나온 공구·장비. '호이스트는 안 씀'처럼 쓰지 않는다고 한 것은 빼고, '고착되면 토치로 가열'처럼 조건부로 쓰게 될 것은 넣습니다.
- materials: 작업 내용에 이름이 나온 화학물질·가스. 추측하지 마세요.

[출력 형식]
{{"summary": "", "equipment": [], "materials": []}}

[예시]
입력: {_EXAMPLE_INPUT}
출력: {json.dumps(_EXAMPLE_OUTPUT, ensure_ascii=False)}
"""


# ── LLM 후보 검증 (코드) ──

_SPACES = re.compile(r"[ \t]+")  # 줄바꿈은 지우지 않는다: 부정 표현이 다음 발화로 넘어가면 안 된다 (text_rules 참고)
_NUMBER = re.compile(r"\d+")
_SPACES_ALL = re.compile(r"\s+")


def _vocab_keys(vocab: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """사전의 모든 표기(공백 제거·소문자) → 표준 이름."""
    return {_SPACES_ALL.sub("", alias).lower(): canonical for canonical, aliases in vocab.items() for alias in aliases}


_EQUIPMENT_KEYS = _vocab_keys(EQUIPMENT)
_MATERIAL_KEYS = _vocab_keys(MATERIALS)
# 물질이 아닌 설비·부품 단어 (임시값. 현장 용어가 쌓이면 늘린다). 장비·장소 사전의 단어는 따로 이미 뺀다.
NON_MATERIAL_WORDS = (
    "플랜지", "볼트", "너트", "파이프", "호스", "용기", "베어링", "축", "기어", "모터", "벨트", "체인", "와이어", "케이블",
    "후크", "가스켓", "패킹", "필터", "센서", "배관", "탱크", "밸브", "펌프", "장비", "설비", "기계", "기구", "공구",
)
_NON_MATERIAL_SUFFIXES = ("탱크", "드럼", "배관", "설비", "장비", "기계", "기구", "용기")  # 예: "톨루엔 탱크"는 설비 이름
_TAG_RE = re.compile(r"^[a-z]{1,4}-?\d{1,4}[a-z]?$|^\d+호기$")  # 설비 번호 (T-105, 3호기)
_NON_MATERIAL_WORDS = (
    frozenset(_EQUIPMENT_KEYS)
    | frozenset(_SPACES_ALL.sub("", w).lower() for w in LOCATION_WORDS | POSITION_WORDS | frozenset(NON_MATERIAL_WORDS))
)


def _compact(value: str) -> str:
    return _SPACES.sub("", value)


def mentioned(text: str, item: str) -> bool:
    """item이 text에 부정되지 않은 채 한 번이라도 나오면 True (공백은 무시한다)."""
    needle = _compact(item).lower()
    haystack = _compact(text).lower()
    if not needle:
        return False
    start = 0
    while (pos := haystack.find(needle, start)) != -1:
        if not is_negated(haystack, pos + len(needle)):
            return True
        start = pos + 1
    return False


def _clean_candidates(values: object, text: str, *, drop: frozenset[str] = frozenset()) -> tuple[list[str], int]:
    """(채택한 후보, 버린 개수). 원문에 없거나 부정된 것, 너무 짧거나 긴 것, drop에 있는 것은 버린다."""
    kept: list[str] = []
    dropped = 0
    for raw in str_list(values)[: MAX_BRIEF_ITEMS * 2]:
        item = " ".join(raw.split())
        key = _compact(item).lower()
        if (
            not MIN_ITEM_CHARS <= len(key) <= MAX_ITEM_CHARS
            or key in drop
            or key.isdigit()
            or not mentioned(text, item)
        ):
            dropped += 1
            continue
        if item not in kept and len(kept) < MAX_BRIEF_ITEMS:
            kept.append(item)
    return kept, dropped


def _canonical(kept: Sequence[str], keys: Mapping[str, str]) -> list[str]:
    """사전에 있는 표기는 표준 이름으로 맞추고(규칙 추출과 같은 이름이라 합칠 때 겹치지 않는다), 아니면 그대로 둔다."""
    return merge([keys.get(_compact(k).lower(), k) for k in kept])


def clean_equipment(values: object, text: str) -> tuple[list[str], int]:
    kept, dropped = _clean_candidates(values, text)
    return _canonical(kept, _EQUIPMENT_KEYS), dropped


def clean_materials(values: object, text: str) -> tuple[list[str], int]:
    """장비·장소 단어는 물질이 아니다. 사전 밖 이름은 MSDS 조회가 '찾지 못함'으로 알려 담당자가 확인하게 한다."""
    kept, dropped = _clean_candidates(values, text, drop=_NON_MATERIAL_WORDS)
    named = [k for k in kept if not _TAG_RE.match(_compact(k).lower()) and not _compact(k).endswith(_NON_MATERIAL_SUFFIXES)]
    return _canonical(named, _MATERIAL_KEYS), dropped + len(kept) - len(named)


def clean_summary(value: object, text: str) -> str | None:
    """한 줄 요약. 비었거나, 원문에 없는 숫자가 들어 있으면 None (지어낸 수치 방지)."""
    if not isinstance(value, str) or len(value.split()) == 0:
        return None
    summary = one_line(value, BRIEF_SUMMARY_CHARS)
    if len(summary) < 4:
        return None
    compact_text = _compact(text)
    if any(n not in compact_text for n in _NUMBER.findall(summary)):
        return None
    return summary


# ── 계획 (LLM 후보 + 코드 검증) ──


class BriefPlanner(Planner):
    """Planner와 같은 plan(work) 규격. LLM은 작업·공구·물질만 내고, 검색 질의는 만들지 않는다(queries=[])."""

    def __init__(self, *, llm: LLMClient, corpora: Mapping[str, Corpus], max_tokens: int = BRIEF_MAX_TOKENS) -> None:
        super().__init__(llm=llm, corpora=corpora, max_tokens=max_tokens)
        self._system = build_brief_prompt()

    def plan(self, work: WorkContext) -> Plan:
        user = build_planner_input(work)
        raw = extract_json_object(self._ask(user))
        if raw is None:
            logger.info("계획 응답 JSON 파싱 실패, 1회 재시도")
            raw = extract_json_object(self._ask(user + _RETRY_SUFFIX))
        if raw is None:
            logger.warning("계획 응답을 읽지 못해 규칙 기반 계획으로 계속합니다.")
            return dataclasses.replace(self._fallback_plan(work), queries=[])

        equipment, eq_dropped = clean_equipment(raw.get("equipment"), work.text)
        materials, mat_dropped = clean_materials(raw.get("materials"), work.text)
        if eq_dropped or mat_dropped:
            logger.info("계획 후보 중 원문에 없거나 부정된 것을 버렸습니다: 공구 %d건, 물질 %d건", eq_dropped, mat_dropped)
        summary = clean_summary(raw.get("summary"), work.text)
        if summary is None:
            logger.info("계획 요약을 쓸 수 없어(비었거나 원문에 없는 숫자) 규칙 요약으로 대신합니다.")
        return Plan(
            summary=summary or one_line(work.text, BRIEF_SUMMARY_CHARS * 2),
            work_types=self._input_work_types(work),  # 작업 유형은 classify_work 결과만 따른다
            materials=merge(str_list(work.nearby_materials), materials),
            equipment=merge(str_list(work.equipment), equipment),
            queries=[],
            source="llm" if summary else "fallback",
        )


# ── 검색 질의 조립 (코드) ──


def msds_query_terms(msds: Sequence[MsdsDto]) -> list[str]:
    """찾은 물질의 GHS 분류 중 사고 검색에 쓸 만한 것 (예: '인화성 액체')."""
    terms: list[str] = []
    for m in msds:
        if m.found:
            terms += [c for c in m.ghs_classes if any(k in c for k in HAZARD_TERM_MARKERS)]
    return merge(terms)[:MAX_HAZARD_TERMS]


def build_queries(
    *,
    summary: str,
    work_types: Sequence[str],
    hazard_terms: Sequence[str],
    requirements: Mapping[str, Sequence[EvidenceRequirement]],
    corpora: Mapping[str, Corpus],
) -> list[SearchQuery]:
    """첫 검색 질의. 작업 유형별 필수 근거마다 하나씩 + 자료 묶음마다 작업 요약 하나.

    사고사례 질의에는 MSDS 위험 분류를 붙인다 (HAZARD_TERM_WORK_TYPES). 물질·장비를 붙인 질의는 일부러 여기에
    쓰지 않는다: 근거가 모자랄 때 재질의(sufficiency.requery)의 첫 변형이 그것이라 겹치지 않게 한다.
    """
    summary_queries = [SearchQuery(name, one_line(summary, MAX_QUERY_CHARS), "작업 요약") for name in corpora] if summary.strip() else []
    room = max(MAX_PLANNED_QUERIES - len(summary_queries), 0)
    queries: list[SearchQuery] = []
    for work_type, req in sufficiency.requirements_for(requirements, work_types):
        text = req.query
        if req.corpus == "accident" and work_type in HAZARD_TERM_WORK_TYPES and hazard_terms:
            text = f"{req.query} {' '.join(hazard_terms)}"
        queries.append(SearchQuery(req.corpus, one_line(text, MAX_QUERY_CHARS), f"필수:{req.key}"))
    return queries[:room] + summary_queries


# ── 수집기: 계획 → 물질 조회 → 검색 ──

MsdsLookup = Callable[[list[str]], list[MsdsDto]]


class StagedCollector(EvidenceCollector):
    """EvidenceCollector에 (1) 계획 LLM 줄이기, (2) 계획 뒤·검색 앞 물질 조회를 끼운 것.

    brief=False, lookup_msds=None 이면 EvidenceCollector와 같은 결과를 낸다.
    msds: collect() 뒤에 lookup_msds를 불렀으면 그 결과(물질이 없으면 []), 안 불렀으면 None.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolInvoker,
        brief: bool = False,
        lookup_msds: MsdsLookup | None = None,
        corpora: Mapping[str, Corpus] | None = None,
        **kwargs,
    ) -> None:
        corpora = dict(corpora if corpora is not None else DEFAULT_CORPORA)
        super().__init__(llm=llm, tools=tools, corpora=corpora, **kwargs)
        self._brief = brief
        self._lookup_msds = lookup_msds
        self._corpora_map = corpora
        self.msds: list[MsdsDto] | None = None
        if brief:
            self._planner = BriefPlanner(llm=llm, corpora=corpora)

    def _with_queries(self, plan: Plan, hazard_terms: Sequence[str]) -> Plan:
        return dataclasses.replace(
            plan,
            queries=build_queries(
                summary=plan.summary,
                work_types=plan.work_types,
                hazard_terms=hazard_terms,
                requirements=self._requirements,
                corpora=self._corpora_map,
            ),
        )

    def collect(self, work: WorkContext) -> EvidencePack:
        started = time.perf_counter()
        plan = self._planner.plan(work)
        if self._brief:
            # 응답 형태를 지키려고 계획 단계 기록에도 이전처럼 질의 목록을 싣는다 (이 시점은 MSDS 위험 분류를 붙이기 전 질의).
            plan = self._with_queries(plan, [])
        self._emitter.emit("plan", {"work": work.model_dump()}, plan.dump(), started)

        self.msds = None
        if self._lookup_msds is not None:
            self.msds = self._lookup_msds(list(plan.materials))
        if self._brief:  # 실제로 검색할 질의: 사고사례 질의에 MSDS 위험 분류가 붙는다 (evidence.search 단계 입력에 남는다)
            plan = self._with_queries(plan, msds_query_terms(self.msds or []))
        return self._collect_with_plan(plan)

    def _collect_with_plan(self, plan: Plan) -> EvidencePack:
        # EvidenceCollector.collect의 ② 이후와 같다 (tests/test_plan_brief.py가 같은 결과임을 확인한다).
        reqs = sufficiency.requirements_for(self._requirements, plan.work_types)
        state = SearchState()
        statuses, rounds = self._search_until_enough(plan, reqs, state)
        if state.all_failed:
            raise ToolError("EVIDENCE_SEARCH_FAILED", "근거 검색이 모두 실패했습니다.")
        return EvidencePack(
            summary=plan.summary,
            work_types=plan.work_types,
            materials=plan.materials,
            equipment=plan.equipment,
            plan_source=plan.source,
            items=state.ranked_items(),
            requirements=statuses,
            missing=[s.description for s in statuses if not s.satisfied],
            search_failures=state.failures,
            rounds=rounds,
        )
