"""① 계획: LLM이 작업·물질·장비를 뽑고 자료 종류별 검색 질의를 만든다.

JSON을 못 읽으면 1회 재시도하고, 그래도 안 되면 규칙 기반 계획(source="fallback")으로 계속한다.
LLM 호출 실패(LLMError)는 그대로 올린다.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse, Message

from .models import SearchQuery, WorkContext
from .textutil import extract_json_object, merge, one_line, str_list
from .vocab import (
    MAX_PLANNED_QUERIES,
    MAX_QUERY_CHARS,
    MAX_SUMMARY_CHARS,
    WORK_TYPES,
    Corpus,
    normalize_work_type,
)

logger = logging.getLogger(__name__)

_RETRY_SUFFIX = "\n\n(이전 응답은 JSON으로 읽을 수 없었습니다. 위 형식의 JSON 객체 하나만 출력하세요.)"

# 예시에는 계획서 9장 데모 문장을 쓰지 않는다 (시연·정답 세트 평가 오염 방지).
_EXAMPLE_INPUT = "작업 내용: 내일 3호기 펌프 베어링 교체. 축이 고착돼 있으면 토치로 가열해서 뺄 거야. 옆에 메탄올 드럼 있음. 2명, 호이스트는 안 씀."
_EXAMPLE_OUTPUT = {
    "summary": "3호기 펌프 베어링 교체, 고착 시 토치 가열",
    "work_types": ["화기"],
    "materials": ["메탄올"],
    "equipment": ["토치"],
    "queries": [
        {"corpus": "accident", "query": "토치 가열 중 인화성 증기 화재 사고", "purpose": "유사 화재 사고사례"},
        {"corpus": "guide", "query": "화기작업 가연성가스 측정 기준", "purpose": "작업 전 가스 측정 기준"},
        {"corpus": "guide", "query": "펌프 정비 회전체 끼임 방지", "purpose": "정비 작업 기준"},
    ],
}


@dataclass
class Plan:
    summary: str
    work_types: list[str]
    materials: list[str]
    equipment: list[str]
    queries: list[SearchQuery]
    source: Literal["llm", "fallback"]

    def dump(self) -> dict:
        """agent_steps 기록용."""
        return {
            "source": self.source,
            "summary": self.summary,
            "work_types": self.work_types,
            "materials": self.materials,
            "equipment": self.equipment,
            "queries": [{"corpus": q.corpus, "query": q.query, "purpose": q.purpose} for q in self.queries],
        }


# ── LLM 메시지 경계 (llm_types와 맞닿는 곳은 여기 두 함수뿐) ──


def _user_message(text: str) -> Message:
    return Message(role="user", content=text)


def response_text(resp: LLMResponse) -> str:
    return resp.text or ""


# ── 프롬프트 ──


def build_planner_prompt(corpora: Mapping[str, Corpus]) -> str:
    corpus_lines = "\n".join(f"  - {name}: {c.description}" for name, c in corpora.items())
    return f"""당신은 산업현장 안전관리 에이전트의 검색 계획 담당입니다.
작업 내용을 읽고, 위험성평가와 작업허가에 필요한 근거를 찾기 위한 검색 계획을 JSON 객체 하나로만 출력하세요. 설명은 쓰지 마세요.

[항목]
- summary: 작업 한 줄 요약 (60자 이내)
- work_types: 작업에 포함된 작업유형. 허용 값: {", ".join(WORK_TYPES)}
  - '호이스트는 안 씀'처럼 부정된 작업은 빼고, '막히면 토치로 가열'처럼 조건부로 하게 될 작업은 넣습니다(안전 쪽으로 판단).
- materials: 작업 내용에 이름이 나온 화학물질·가스. 추측하지 마세요.
- equipment: 작업 내용에 나온 공구·장비.
- queries: 검색 질의 2~6개. 각 질의는 corpus, query, purpose를 가집니다.
  - corpus는 아래 목록의 이름만 씁니다.
{corpus_lines}
  - query는 40자 이내의 짧은 검색 문장입니다. 작업 내용에 없는 사실(설비명, 수치)을 지어내지 마세요.
  - purpose는 이 질의로 무엇을 찾으려는지 짧게 씁니다.

[출력 형식]
{{"summary": "", "work_types": [], "materials": [], "equipment": [], "queries": [{{"corpus": "", "query": "", "purpose": ""}}]}}

[예시]
입력: {_EXAMPLE_INPUT}
출력: {json.dumps(_EXAMPLE_OUTPUT, ensure_ascii=False)}
"""


def build_planner_input(work: WorkContext) -> str:
    lines = [f"작업 내용: {work.text.strip()}"]
    if work.location:
        lines.append(f"장소: {work.location}")
    if work.hot_work_kind:
        lines.append(f"화기 종류: {work.hot_work_kind}")
    if work.nearby_materials:
        lines.append(f"주변 물질: {', '.join(work.nearby_materials)}")
    if work.equipment:
        lines.append(f"투입 장비: {', '.join(work.equipment)}")
    if work.headcount:
        lines.append(f"인원: {work.headcount}명")
    if work.end_time:
        lines.append(f"종료 시간: {work.end_time}")
    return "\n".join(lines)


# ── 계획 만들기 ──


class Planner:
    def __init__(self, *, llm: LLMClient, corpora: Mapping[str, Corpus], max_tokens: int) -> None:
        self._llm = llm
        self._corpora = corpora
        self._max_tokens = max_tokens
        self._system = build_planner_prompt(corpora)

    def plan(self, work: WorkContext) -> Plan:
        user = build_planner_input(work)
        raw = extract_json_object(self._ask(user))
        if raw is None:
            logger.info("계획 응답 JSON 파싱 실패, 1회 재시도")
            raw = extract_json_object(self._ask(user + _RETRY_SUFFIX))
        if raw is None:
            logger.warning("계획 응답을 읽지 못해 규칙 기반 계획으로 계속합니다.")
            return self._fallback_plan(work)
        return self._parse_plan(raw, work)

    def _ask(self, user: str) -> str:
        resp = self._llm.chat([_user_message(user)], system=self._system, max_tokens=self._max_tokens)
        if resp.stop_reason == "max_tokens":
            logger.info("계획 응답이 max_tokens에서 잘림 (max_tokens=%s)", self._max_tokens)
        return response_text(resp)

    @staticmethod
    def _input_work_types(work: WorkContext) -> list[str]:
        return merge([w for w in (normalize_work_type(v) for v in work.work_types) if w])

    def _parse_plan(self, raw: dict, work: WorkContext) -> Plan:
        summary = raw.get("summary")
        summary = (
            one_line(summary, MAX_SUMMARY_CHARS)
            if isinstance(summary, str) and summary.strip()
            else one_line(work.text, MAX_SUMMARY_CHARS)
        )

        planned_types = [w for w in (normalize_work_type(v) for v in str_list(raw.get("work_types"))) if w]
        # 입력(classify_work 결과)과 계획 결과를 합친다. 필수 근거가 늘어나는 쪽(안전 쪽)으로 판단한다.
        work_types = merge(self._input_work_types(work), planned_types)
        if len(work_types) > 1 and "일반" in work_types:
            work_types.remove("일반")

        queries = self._parse_queries(raw.get("queries")) or self._base_queries(summary)
        return Plan(
            summary=summary,
            work_types=work_types,
            materials=merge(str_list(work.nearby_materials), str_list(raw.get("materials"))),
            equipment=merge(str_list(work.equipment), str_list(raw.get("equipment"))),
            queries=queries,
            source="llm",
        )

    def _parse_queries(self, raw_queries: object) -> list[SearchQuery]:
        """자료 종류를 모르거나 문장이 없는 질의는 버리고, 같은 질의는 하나로 합친다."""
        if not isinstance(raw_queries, list):
            return []
        queries: list[SearchQuery] = []
        seen: set[tuple[str, str]] = set()
        for q in raw_queries:
            if not isinstance(q, dict):
                continue
            corpus, text = q.get("corpus"), q.get("query")
            if corpus not in self._corpora or not isinstance(text, str) or not text.strip():
                continue
            text = one_line(text, MAX_QUERY_CHARS)
            purpose = q.get("purpose") if isinstance(q.get("purpose"), str) and q.get("purpose").strip() else "계획 질의"
            if (corpus, text) in seen:
                continue
            seen.add((corpus, text))
            queries.append(SearchQuery(corpus, text, one_line(purpose, 60)))
            if len(queries) == MAX_PLANNED_QUERIES:
                break
        return queries

    def _base_queries(self, summary: str) -> list[SearchQuery]:
        text = one_line(summary, MAX_QUERY_CHARS)
        return [SearchQuery(name, text, "기본 질의") for name in self._corpora]

    def _fallback_plan(self, work: WorkContext) -> Plan:
        summary = one_line(work.text, MAX_SUMMARY_CHARS)
        return Plan(
            summary=summary,
            work_types=self._input_work_types(work),
            materials=merge(str_list(work.nearby_materials)),
            equipment=merge(str_list(work.equipment)),
            queries=self._base_queries(summary),
            source="fallback",
        )
