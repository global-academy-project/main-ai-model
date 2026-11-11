"""계획 줄이기(brief)와 계획서 순서(계획 → 물질 조회 → 근거 검색) 시험 (app/agent/plan_brief.py).

모델은 부르지 않고 가짜 LLM을 쓴다.
검증 범위: 가짜 LLM + 전체 app 코드 실행 + mock 도구 (실제 모델·검색 품질은 확인하지 않는다).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("app.agent.work_register")  # 앱 코드가 없는 환경(zip만 푼 폴더 포함)에서는 건너뛴다

from app.agent import plan_brief, work_register  # noqa: E402
from app.agent.evidence_collector import EvidenceCollector, WorkContext  # noqa: E402
from app.agent.evidence_collector.vocab import DEFAULT_CORPORA, DEFAULT_REQUIREMENTS, MAX_PLANNED_QUERIES  # noqa: E402
from app.agent.plan_brief import (  # noqa: E402
    BRIEF_MAX_TOKENS,
    BriefPlanner,
    StagedCollector,
    build_brief_prompt,
    build_queries,
    clean_equipment,
    clean_materials,
    clean_summary,
    mentioned,
    msds_query_terms,
)
from app.exceptions import LLMError  # noqa: E402
from app.llm.llm_base import LLMClient  # noqa: E402
from app.llm.llm_types import LLMResponse  # noqa: E402
from app.schemas.agent_work import MsdsDto, WorkRegisterRequest  # noqa: E402
from app.tools.tool_setup import build_tool_registry  # noqa: E402


class StubLLM(LLMClient):
    name = "STUB"

    def __init__(self, plan_text: str = "", *, error: Exception | None = None) -> None:
        super().__init__(None)
        self.plan_text = plan_text
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        kind = "pick" if "근거 선택 담당" in system else "judge" if "위험요인 판단 담당" in system else "plan"
        self.calls.append({"kind": kind, "max_tokens": max_tokens, "user": messages[-1].content, "system": system})
        if kind == "plan" and self.error is not None:
            raise self.error
        return LLMResponse(text=self.plan_text if kind == "plan" else "모름", stop_reason="end")

    def kinds(self) -> list[str]:
        return [c["kind"] for c in self.calls]


def _brief(summary="3호기 배관 용접", equipment=(), materials=()) -> str:
    return json.dumps({"summary": summary, "equipment": list(equipment), "materials": list(materials)}, ensure_ascii=False)


_TEXT = "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 옆 T-105 탱크에 톨루엔. 3명, 15시까지. 크레인은 안 씀."


# ── 후보 검증 ──


def test_mentioned_ignores_spaces_and_case():
    assert mentioned("고소 작업대를 씁니다", "고소작업대")
    assert mentioned("ipa 세척", "IPA")
    assert not mentioned("그라인더 사용", "용접기")


@pytest.mark.parametrize("text", ["크레인은 안 씀", "크레인 사용 안 함", "크레인 없음", "크레인은안씀"])
def test_mentioned_rejects_negated(text):
    assert not mentioned(text, "크레인")


def test_mentioned_accepts_when_any_mention_is_not_negated():
    assert mentioned("크레인은 안 씀\n오후에는 크레인으로 인양", "크레인")


def test_mentioned_negation_does_not_cross_lines():
    assert mentioned("크레인 인양\n안 쓰는 건 없음", "크레인")


def test_clean_equipment_keeps_only_mentioned_and_not_negated():
    kept, dropped = clean_equipment(["그라인더", "크레인", "용접기", "그라 인더"], _TEXT)
    assert kept == ["그라인더"] and dropped == 2  # 크레인(부정)·용접기(없음). "그라 인더"는 공백만 다른 같은 말이라 합친다


def test_clean_equipment_uses_canonical_names_and_keeps_unknown_mentioned_ones():
    kept, _ = clean_equipment(["스카이차", "플라즈마 절단기"], "스카이차로 올라가 플라즈마 절단기로 자름")
    assert kept == ["고소작업대", "플라즈마 절단기"]  # 사전의 별칭은 표준 이름, 사전 밖 이름은 원문 표기


def test_clean_equipment_caps_count_and_length():
    names = [f"장비{i}번" for i in range(20)]
    kept, _ = clean_equipment(names, " ".join(names))
    assert len(kept) == plan_brief.MAX_BRIEF_ITEMS
    assert clean_equipment(["가" * 40], "가" * 40) == ([], 1)
    assert clean_equipment(["x"], "x") == ([], 1)  # 한 글자


def test_clean_equipment_non_list_input_is_empty():
    assert clean_equipment(None, _TEXT) == ([], 0)
    assert clean_equipment({"a": 1}, _TEXT) == ([], 0)
    assert clean_equipment("그라인더", _TEXT) == (["그라인더"], 0)  # 문자열 하나는 목록으로 본다


def test_clean_materials_drops_tags_equipment_places_and_parts():
    kept, dropped = clean_materials(["톨루엔", "T-105", "그라인더", "배관", "플랜지", "볼트", "크레인"], _TEXT)
    assert kept == ["톨루엔"] and dropped == 6


def test_clean_materials_drops_facility_names_ending_with_tank_etc():
    kept, _ = clean_materials(["톨루엔 탱크", "톨루엔"], "톨루엔 탱크 옆에서 작업. 톨루엔 냄새")
    assert kept == ["톨루엔"]


def test_clean_materials_canonicalizes_and_keeps_unknown_chemicals_for_msds_check():
    kept, _ = clean_materials(["크실렌", "에틸렌글리콜"], "크실렌 통과 에틸렌글리콜 통 옆")
    assert kept == ["자일렌", "에틸렌글리콜"]  # 사전 밖 이름은 MSDS 조회가 '찾지 못함'으로 알린다


def test_clean_materials_drops_unmentioned_guess():
    assert clean_materials(["벤젠"], _TEXT) == ([], 1)


@pytest.mark.parametrize("value", [None, "", "  ", 3, ["a"], "가나"])
def test_clean_summary_rejects_unusable(value):
    assert clean_summary(value, _TEXT) is None


def test_clean_summary_rejects_numbers_not_in_text_and_keeps_numbers_in_text():
    assert clean_summary("R-201 반응기 하부 배관 플랜지 교체, 7명 투입", _TEXT) is None  # 7명은 원문에 없다
    assert clean_summary("R-201 반응기 하부 배관 플랜지 볼트 교체", _TEXT) == "R-201 반응기 하부 배관 플랜지 볼트 교체"


def test_clean_summary_is_one_line_and_cut():
    out = clean_summary("가나다라\n" + "마" * 200, "가나다라 " + "마" * 200)
    assert "\n" not in out and len(out) == plan_brief.BRIEF_SUMMARY_CHARS


# ── 프롬프트 ──


def test_prompt_asks_only_for_summary_equipment_materials_and_has_no_demo_sentence():
    prompt = build_brief_prompt()
    assert '"summary"' in prompt and '"equipment"' in prompt and '"materials"' in prompt
    assert "queries" not in prompt and "work_types" not in prompt
    for demo in ("R-201", "T-105", "플랜지", "13시"):  # 계획서 9장 데모 문장 오염 방지
        assert demo not in prompt


# ── BriefPlanner ──


def _work(**kw) -> WorkContext:
    base = dict(text=_TEXT, work_types=["화기"], nearby_materials=[], equipment=[])
    base.update(kw)
    return WorkContext(**base)


def _planner(llm) -> BriefPlanner:
    return BriefPlanner(llm=llm, corpora=DEFAULT_CORPORA)


def test_planner_success_one_call_with_small_token_budget():
    llm = StubLLM(_brief("R-201 반응기 하부 배관 플랜지 볼트 교체", ["그라인더", "크레인"], ["톨루엔", "T-105"]))
    plan = _planner(llm).plan(_work())
    assert llm.kinds() == ["plan"] and llm.calls[0]["max_tokens"] == BRIEF_MAX_TOKENS
    assert plan.source == "llm" and plan.queries == []
    assert plan.summary == "R-201 반응기 하부 배관 플랜지 볼트 교체"
    assert plan.equipment == ["그라인더"] and plan.materials == ["톨루엔"]


def test_planner_work_types_come_only_from_classification():
    raw = json.dumps({"summary": "배관 플랜지 볼트 교체", "equipment": [], "materials": [], "work_types": ["고소", "전기"]}, ensure_ascii=False)
    plan = _planner(StubLLM(raw)).plan(_work(work_types=["화기"]))
    assert plan.work_types == ["화기"]
    assert _planner(StubLLM(raw)).plan(_work(work_types=[])).work_types == []


def test_planner_merges_slot_values_first_without_duplicates():
    llm = StubLLM(_brief("배관 플랜지 볼트 교체", ["그라인더"], ["톨루엔"]))
    plan = _planner(llm).plan(_work(nearby_materials=["톨루엔", "메탄올"], equipment=["그라인더", "용접기"]))
    assert plan.materials == ["톨루엔", "메탄올"] and plan.equipment == ["그라인더", "용접기"]


def test_planner_unreadable_retries_once_then_falls_back():
    llm = StubLLM("모름")
    plan = _planner(llm).plan(_work(nearby_materials=["톨루엔"], equipment=["그라인더"]))
    assert llm.kinds() == ["plan", "plan"]
    assert plan.source == "fallback" and plan.queries == []
    assert plan.materials == ["톨루엔"] and plan.equipment == ["그라인더"] and plan.work_types == ["화기"]


def test_planner_retry_can_succeed():
    answers = iter(["모름", _brief("배관 플랜지 볼트 교체", ["그라인더"], [])])

    class Once(StubLLM):
        def chat(self, messages, *, system="", tools=None, max_tokens=None):
            self.calls.append({"kind": "plan", "max_tokens": max_tokens, "user": messages[-1].content, "system": system})
            return LLMResponse(text=next(answers), stop_reason="end")

    llm = Once()
    plan = _planner(llm).plan(_work())
    assert len(llm.calls) == 2 and "JSON" in llm.calls[1]["user"] and plan.source == "llm"


def test_planner_unusable_summary_marks_fallback_but_keeps_candidates():
    plan = _planner(StubLLM(_brief("", ["그라인더"], ["톨루엔"]))).plan(_work())
    assert plan.source == "fallback"  # 요약이 없으면 규칙 요약을 쓰게 한다 (work_register)
    assert plan.equipment == ["그라인더"] and plan.materials == ["톨루엔"]


def test_planner_llm_error_is_not_swallowed():
    with pytest.raises(LLMError):
        _planner(StubLLM(error=LLMError("실패"))).plan(_work())


# ── MSDS 분류 · 질의 조립 ──


def _msds(name, classes, found=True) -> MsdsDto:
    return MsdsDto(query=name, found=found, name=name if found else None, ghs_classes=classes)


def test_msds_query_terms_use_only_found_relevant_classes_and_cap():
    msds = [_msds("톨루엔", ["인화성 액체", "흡인 유해성"]), _msds("가스", ["인화성 가스", "고압가스"]), _msds("없음", ["인화성 액체"], found=False)]
    assert msds_query_terms(msds) == ["인화성 액체", "인화성 가스"]
    assert msds_query_terms([]) == []
    assert msds_query_terms([_msds("x", ["흡인 유해성"])]) == []
    assert msds_query_terms([_msds("없음", ["부식성"], found=False)]) == []  # 못 찾은 물질의 분류는 쓰지 않는다


def _q(work_types, terms=("인화성 액체",), summary="배관 플랜지 교체"):
    return build_queries(summary=summary, work_types=work_types, hazard_terms=list(terms), requirements=DEFAULT_REQUIREMENTS, corpora=DEFAULT_CORPORA)


def test_build_queries_hot_work_demo_shape():
    queries = _q(["화기"])
    by_purpose = {q.purpose: q for q in queries}
    assert by_purpose["필수:fire.accidents"].query == "화기작업 화재 폭발 사고 사례 인화성 액체"
    assert by_purpose["필수:fire.accidents"].corpus == "accident"
    assert by_purpose["필수:fire.gas_check"].query == "화기작업 가연성가스 농도 측정 기준"  # 기준 질의에는 붙이지 않는다
    assert by_purpose["필수:fire.watch"].corpus == "guide"
    assert [(q.corpus, q.query) for q in queries if q.purpose == "작업 요약"] == [("accident", "배관 플랜지 교체"), ("guide", "배관 플랜지 교체")]


def test_build_queries_hazard_terms_only_for_hot_work_and_confined_space():
    height = {q.purpose: q.query for q in _q(["고소"])}
    assert height["필수:height.accidents"] == "고소작업 추락 사고 사례"
    confined = {q.purpose: q.query for q in _q(["밀폐공간"], terms=("고압가스",))}
    assert confined["필수:confined.accidents"].endswith("고압가스")


def test_build_queries_without_hazard_terms_or_summary():
    assert {q.purpose: q.query for q in _q(["화기"], terms=())}["필수:fire.accidents"] == "화기작업 화재 폭발 사고 사례"
    assert not [q for q in _q(["화기"], summary="  ") if q.purpose == "작업 요약"]


def test_build_queries_general_work_has_only_summary_queries():
    assert [q.purpose for q in _q(["일반"])] == ["작업 요약", "작업 요약"]
    assert [q.purpose for q in _q([])] == ["작업 요약", "작업 요약"]


def test_build_queries_cap_keeps_summary_queries():
    queries = _q(["화기", "밀폐공간", "고소", "전기", "중량물"])
    assert len(queries) == MAX_PLANNED_QUERIES
    assert [q.purpose for q in queries[-2:]] == ["작업 요약", "작업 요약"]


def test_build_queries_are_unique_and_short():
    queries = _q(["화기"], summary="가" * 500)
    assert len({(q.corpus, q.query) for q in queries}) == len(queries)
    assert all(len(q.query) <= 100 for q in queries)


# ── StagedCollector ──


def _dump(pack):
    return pack.model_dump()


def _collect(collector_cls, llm, **kw):
    steps = []
    collector = collector_cls(llm=llm, tools=build_tool_registry(), on_step=steps.append, **kw)
    work = WorkContext(text="3호기 배관 용접. 옆에 메탄올 드럼.", nearby_materials=["메탄올"], equipment=["용접기"], work_types=["화기"])
    return collector, collector.collect(work), steps


def _stable(steps):
    return [(s.name, s.input, s.output) for s in steps]


def test_staged_collector_default_equals_evidence_collector():
    """brief=False, lookup=None이면 변경 전 EvidenceCollector와 같은 근거 묶음·단계 기록."""
    plan_json = json.dumps({"summary": "배관 용접", "work_types": ["화기", "고소"], "materials": ["메탄올"], "equipment": ["용접기"],
                            "queries": [{"corpus": "accident", "query": "용접 화재 사고", "purpose": "사고"}, {"corpus": "guide", "query": "화기작업 기준", "purpose": "기준"}]}, ensure_ascii=False)
    _, old_pack, old_steps = _collect(EvidenceCollector, StubLLM(plan_json))
    staged, new_pack, new_steps = _collect(StagedCollector, StubLLM(plan_json))
    assert _dump(new_pack) == _dump(old_pack)
    assert _stable(new_steps) == _stable(old_steps)
    assert staged.msds is None


def test_staged_collector_default_equals_evidence_collector_when_llm_unreadable():
    _, old_pack, old_steps = _collect(EvidenceCollector, StubLLM("모름"))
    _, new_pack, new_steps = _collect(StagedCollector, StubLLM("모름"))
    assert _dump(new_pack) == _dump(old_pack) and _stable(new_steps) == _stable(old_steps)


def test_staged_collector_looks_up_msds_between_plan_and_search():
    events: list[str] = []
    calls: list[list[str]] = []

    def lookup(materials):
        events.append("lookup")
        calls.append(list(materials))
        return [_msds("메탄올", ["인화성 액체"])]

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def call(self, name, arguments):
            events.append(name)
            return self.inner.call(name, arguments)

    steps = []
    collector = StagedCollector(llm=StubLLM(_brief("3호기 배관 용접", ["용접기"], ["메탄올"])), tools=Spy(build_tool_registry()),
                                on_step=lambda s: (events.append(f"step:{s.name}"), steps.append(s)), brief=True, lookup_msds=lookup)
    work = WorkContext(text="3호기 배관 용접. 용접기로 작업. 옆에 메탄올 드럼.", work_types=["화기"])
    pack = collector.collect(work)
    assert events[:2] == ["step:plan", "lookup"] and events[2] in ("search_accidents", "search_guides")
    assert calls == [["메탄올"]] and [m.query for m in collector.msds] == ["메탄올"]
    assert pack.materials == ["메탄올"] and pack.plan_source == "llm"
    searched = next(s for s in steps if s.name == "search").input["queries"]
    assert any(q["query"] == "화기작업 화재 폭발 사고 사례 인화성 액체" and q["tool"] == "search_accidents" for q in searched)  # MSDS 분류가 질의에 반영된다
    planned = next(s for s in steps if s.name == "plan").output["queries"]
    assert any(q["query"] == "화기작업 화재 폭발 사고 사례" for q in planned)  # 계획 단계 기록은 MSDS 분류를 붙이기 전 질의


def test_staged_collector_brief_builds_queries_in_code_even_when_llm_unreadable():
    staged, pack, steps = _collect(StagedCollector, StubLLM("모름"), brief=True)
    assert pack.plan_source == "fallback" and staged.msds is None
    plan_step = next(s for s in steps if s.name == "plan")
    planned = plan_step.output["queries"]  # 이전과 같은 형태: {corpus, query, purpose} 목록 (코드가 만든 질의)
    assert planned and all(set(q) == {"corpus", "query", "purpose"} for q in planned)
    searched = next(s for s in steps if s.name == "search").input["queries"]
    assert {q["purpose"] for q in searched} >= {"필수:fire.accidents", "필수:fire.gas_check", "필수:fire.watch"}
    assert {(q["corpus"], q["query"]) for q in planned} == {(q["corpus"], q["query"]) for q in searched}  # MSDS 분류가 없으면 계획 = 검색


def test_staged_collector_lookup_receives_plan_materials_and_result_is_kept():
    seen = []
    staged, pack, _ = _collect(StagedCollector, StubLLM("모름"), brief=True, lookup_msds=lambda m: seen.append(m) or [])
    assert seen == [["메탄올"]] and pack.materials == ["메탄올"]
    assert staged.msds == []  # 조회했지만 결과가 없음 (None은 "조회하지 않았다")


def test_staged_collector_all_searches_failing_raises_like_before():
    from app.exceptions import ToolError

    class Failing:
        def call(self, name, arguments):
            raise ToolError("SEARCH_DOWN", "검색 불가", tool=name)

    for cls, kw in ((EvidenceCollector, {}), (StagedCollector, {"brief": True})):
        collector = cls(llm=StubLLM(_brief("3호기 배관 용접")), tools=Failing(), **kw)
        with pytest.raises(ToolError) as exc:
            collector.collect(WorkContext(text="3호기 배관 용접", work_types=["화기"]))
        assert exc.value.code == "EVIDENCE_SEARCH_FAILED"


# ── work_register 연결 ──

_FIRST = "오늘 13시에 화기작업 있어. 필요한 거 만들어줘."
_SECOND = _TEXT.replace("15시까지", "13시부터 15시까지")


def _run(llm, *, mode="brief", msds_first=True, monkeypatch, message=None, slots=None):
    monkeypatch.setattr(work_register, "PLAN_MODE", mode)
    monkeypatch.setattr(work_register, "MSDS_FIRST", msds_first)
    agent = work_register.WorkRegisterAgent(llm=llm, tools=build_tool_registry())
    return agent.run(WorkRegisterRequest(message=message or _SECOND, run_id=1, site_id=1, slots=slots))


def _names(result):
    return [s.name for s in result.steps]


def test_defaults_are_brief_and_msds_first():
    assert plan_brief.PLAN_MODE == "brief" and work_register.PLAN_MODE == "brief"
    assert plan_brief.MSDS_FIRST is True and work_register.MSDS_FIRST is True


def test_agent_order_follows_the_plan_document(monkeypatch):
    llm = StubLLM(_brief("R-201 반응기 하부 배관 플랜지 볼트 교체", ["그라인더", "크레인"], ["톨루엔"]))
    result = _run(llm, monkeypatch=monkeypatch)
    names = _names(result)
    assert result.run_status == "DONE"
    assert names.index("classify_work") < names.index("evidence.plan") < names.index("lookup_msds") < names.index("search_accidents")
    assert names.index("lookup_msds") < names.index("search_guides") < names.index("check_rules") < names.index("fill_form")
    assert llm.kinds().count("plan") == 1 and llm.calls[0]["max_tokens"] == BRIEF_MAX_TOKENS
    assert result.llm_sources.plan == "llm"
    assert [m.query for m in result.msds] == ["톨루엔"]
    plan_out = next(s for s in result.steps if s.name == "evidence.plan").output
    assert plan_out["equipment"] == ["그라인더"] and plan_out["materials"] == ["톨루엔"]  # 크레인은 부정돼 빠진다


def test_agent_brief_queries_carry_msds_hazard_class(monkeypatch):
    result = _run(StubLLM(_brief("플랜지 볼트 교체", ["그라인더"], ["톨루엔"])), monkeypatch=monkeypatch)
    search = next(s for s in result.steps if s.name == "evidence.search")
    assert any(q["tool"] == "search_accidents" and q["query"] == "화기작업 화재 폭발 사고 사례 인화성 액체" for q in search.input["queries"])


def test_agent_brief_garbage_llm_still_completes_with_code_queries(monkeypatch):
    llm = StubLLM("모름")
    result = _run(llm, monkeypatch=monkeypatch)
    names = _names(result)
    assert result.run_status == "DONE" and result.llm_sources.plan == "fallback"
    assert names.index("evidence.plan") < names.index("lookup_msds") < names.index("search_accidents")
    assert llm.kinds().count("plan") == 2  # 이전처럼 한 번 더 시도


def test_agent_llm_error_in_plan_stops_the_run_as_before(monkeypatch):
    from app.exceptions import AgentRunError

    with pytest.raises(AgentRunError):
        _run(StubLLM(error=LLMError("실패")), monkeypatch=monkeypatch)


def test_agent_waiting_input_never_reaches_plan_or_msds(monkeypatch):
    llm = StubLLM(_brief())
    result = _run(llm, monkeypatch=monkeypatch, message=_FIRST)
    assert result.run_status == "WAITING_INPUT"
    assert llm.calls == [] and _names(result) == ["classify_work"]


def test_agent_unknown_llm_material_is_flagged_by_msds(monkeypatch):
    message = _SECOND + " 에틸렌글리콜 통도 있음."
    result = _run(StubLLM(_brief("배관 플랜지 볼트 교체", ["그라인더"], ["톨루엔", "에틸렌글리콜"])), monkeypatch=monkeypatch, message=message)
    assert [m.query for m in result.msds] == ["톨루엔", "에틸렌글리콜"]
    assert any(i.code == "MSDS_NOT_FOUND" and "에틸렌글리콜" in i.message for i in result.verification)


def _without_time(result):
    dump = result.model_dump(by_alias=True)
    for step in dump["steps"]:
        step["elapsedMs"] = 0
    return dump


class _OldStyleCollector(EvidenceCollector):
    """변경 전 동작: StagedCollector 자리에 EvidenceCollector를 그대로 쓴다 (brief·lookup_msds는 무시)."""

    msds = None

    def __init__(self, *, brief=False, lookup_msds=None, **kw):
        super().__init__(**kw)


def test_off_switches_give_the_same_result_as_before(monkeypatch):
    plan_json = json.dumps({"summary": "배관 용접", "work_types": ["화기", "고소"], "materials": ["톨루엔", "에틸렌글리콜"], "equipment": ["그라인더"],
                            "queries": [{"corpus": "accident", "query": "그라인더 화재 사고", "purpose": "사고"}]}, ensure_ascii=False)
    for plan_text in (plan_json, "모름"):
        new = _without_time(_run(StubLLM(plan_text), mode="full", msds_first=False, monkeypatch=monkeypatch))
        monkeypatch.setattr(work_register, "StagedCollector", _OldStyleCollector)
        old = _without_time(_run(StubLLM(plan_text), mode="full", msds_first=False, monkeypatch=monkeypatch))
        monkeypatch.undo()
        assert new == old


def _key_paths(value, prefix=""):
    """응답 JSON의 키 경로 집합 (값은 보지 않는다). 목록은 [] 로 합친다."""
    paths = set()
    if isinstance(value, dict):
        for key, item in value.items():
            paths |= {f"{prefix}/{key}"} | _key_paths(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for item in value:
            paths |= _key_paths(item, f"{prefix}[]")
    return paths


@pytest.mark.parametrize("plan_text", [_brief("R-201 반응기 하부 배관 플랜지 볼트 교체", ["그라인더"], ["톨루엔"]), "모름"])
def test_response_json_shape_is_the_same_as_before(monkeypatch, plan_text):
    """응답 JSON의 키 구조(최상위 키 순서, 중첩 키 경로)가 변경 전 방식과 같다. 값(단계 순서·질의 문장 등)은 다를 수 있다."""
    new = _without_time(_run(StubLLM(plan_text), mode="brief", msds_first=True, monkeypatch=monkeypatch))
    monkeypatch.setattr(work_register, "StagedCollector", _OldStyleCollector)
    old = _without_time(_run(StubLLM(plan_text), mode="full", msds_first=False, monkeypatch=monkeypatch))
    assert list(new) == list(old)
    assert _key_paths(new) == _key_paths(old)
    assert {s["name"] for s in new["steps"]} == {s["name"] for s in old["steps"]}  # 단계 이름 종류도 같다


def test_msds_first_only_keeps_verification_order_and_results(monkeypatch):
    """MSDS_FIRST만 켜도(full 계획) 결과 내용·확인 항목 순서는 같고 단계 순서만 바뀐다."""
    plan_json = json.dumps({"summary": "배관 용접", "work_types": ["화기", "고소"], "materials": ["톨루엔", "에틸렌글리콜"], "equipment": ["그라인더"],
                            "queries": [{"corpus": "accident", "query": "그라인더 화재 사고", "purpose": "사고"}]}, ensure_ascii=False)
    before = _run(StubLLM(plan_json), mode="full", msds_first=False, monkeypatch=monkeypatch)
    after = _run(StubLLM(plan_json), mode="full", msds_first=True, monkeypatch=monkeypatch)
    assert [(i.code, i.message) for i in before.verification] == [(i.code, i.message) for i in after.verification]
    assert any(i.code == "PLAN_EXTRA_WORK_TYPE" for i in after.verification)
    assert before.msds == after.msds and before.drafts == after.drafts
    names = _names(after)
    assert names.index("lookup_msds") < names.index("search_accidents")
    assert _names(before).index("lookup_msds") > _names(before).index("search_accidents")


# ── 적용 스크립트 (apply_plan_brief.py) ──


def _load_apply_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "apply_plan_brief.py"
    if not path.is_file():
        pytest.skip("apply_plan_brief.py가 저장소 루트에 없습니다")
    spec = importlib.util.spec_from_file_location("apply_plan_brief", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unpatched_text(script) -> str:
    return (
        "import x\n"
        + script.IMPORT_OLD
        + "from .judgment import judge\n\n"
        + "def run():\n"
        + script.COLLECTOR_OLD
        + "    pack = collector.collect(work)\n"
        + script.MSDS_OLD
        + "    return msds\n"
    )


@pytest.fixture
def script_env(tmp_path, monkeypatch):
    script = _load_apply_script()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app" / "agent").mkdir(parents=True)
    return script, tmp_path / "app" / "agent" / "work_register.py"


@pytest.mark.parametrize("newline", ["\r\n", "\n"])
def test_apply_script_apply_is_idempotent_and_revert_restores_bytes(script_env, newline):
    script, target = script_env
    original = _unpatched_text(script).replace("\n", newline).encode("utf-8")
    target.write_bytes(original)
    assert script.main([]) == 0
    patched = target.read_bytes()
    assert patched != original and b"StagedCollector" in patched
    assert (b"\r\n" in patched) == (newline == "\r\n")  # 줄바꿈 유지
    added = sum(len(new.splitlines()) - len(old.splitlines()) for old, new in script.PAIRS)
    assert patched.replace(b"\r\n", b"\n").count(b"\n") == original.replace(b"\r\n", b"\n").count(b"\n") + added
    assert script.main([]) == 0 and target.read_bytes() == patched  # 두 번째는 바꾸지 않는다
    assert script.main(["--check"]) == 0
    assert script.main(["--revert"]) == 0
    assert target.read_bytes() == original


def test_apply_script_stops_without_changing_when_an_anchor_is_missing(script_env):
    script, target = script_env
    text = _unpatched_text(script).replace(script.MSDS_OLD, "    msds = other()\n").encode("utf-8")
    target.write_bytes(text)
    assert script.main([]) == 1
    assert target.read_bytes() == text


def test_apply_script_stops_when_an_anchor_appears_twice(script_env):
    script, target = script_env
    text = (_unpatched_text(script) + script.MSDS_OLD).encode("utf-8")
    target.write_bytes(text)
    assert script.main([]) == 1
    assert target.read_bytes() == text


def test_apply_script_stops_when_run_outside_repo_root(script_env):
    script, target = script_env  # 파일을 만들지 않았다
    assert not target.exists()
    assert script.main([]) == 2
