"""판단 '코드' 방식(app/agent/judgment_code.py) 시험. 모델은 부르지 않고, 검색 도구는 가짜 또는 mock을 쓴다.

검증 범위: 가짜 도구·가짜 LLM + 전체 app 코드 실행 (실제 모델·실제 RAG 검색 품질은 확인하지 않는다).
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("app.agent.work_register")  # 앱 코드가 없는 환경에서는 건너뛴다

from app.agent import judgment_code, work_register  # noqa: E402
from app.agent.evidence_collector import EvidenceItem, EvidencePack
from app.agent.judgment import JudgeInput, rule_hazards
from app.agent.judgment_code import (
    JUDGMENT_CODE,
    MAX_CITATIONS,
    MIN_SCORE,
    SPECS,
    hazard_queries,
    judge_code,
    score,
    select_citations,
)
from app.exceptions import ToolError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse
from app.schemas.agent_work import MsdsDto, WorkRegisterRequest
from app.tools.tool_setup import build_tool_registry


def _item(n: int, text: str, source: str | None = None, corpus: str = "guide") -> EvidenceItem:
    return EvidenceItem(id=f"id-{n}", doc_id="UNKNOWN", source=source or f"출처{n}", text=text, score=1.0 / n, corpus=corpus, round=1)


def _hit(n: int, text: str, source: str | None = None) -> dict:
    return {"id": f"hit-{n}", "doc_id": "UNKNOWN", "source": source or f"새출처{n}", "text": text, "score": 1.0 / n}


def _inp(items=None, *, msds=None, work_types=("화기",), kind="용접", equipment=("용접기",), text="3호기 배관 용접 작업") -> JudgeInput:
    pack = EvidencePack(
        summary="요약", work_types=list(work_types), materials=[], equipment=list(equipment), plan_source="fallback",
        items=items if items is not None else [], requirements=[], missing=[], search_failures=[], rounds=1,
    )
    return JudgeInput(
        text=text, summary="요약", work_types=list(work_types), location="3호기 배관", hot_work_kind=kind,
        materials=[], equipment=list(equipment), pack=pack, msds=msds or [],
    )


class FakeTools:
    """search_guides·search_accidents 호출을 기록하고 질의에 따라 정해 둔 결과를 돌려준다."""

    def __init__(self, guides=None, accidents=None, *, error: Exception | None = None, raw=None) -> None:
        self.guides = guides or []
        self.accidents = accidents or []
        self.error = error
        self.raw = raw
        self.calls: list[tuple[str, dict]] = []

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.error is not None:
            raise self.error
        if self.raw is not None:
            return self.raw
        return {"hits": self.guides if name == "search_guides" else self.accidents}


FIRE_GUIDE = "산업안전보건기준 제241조의2 용접·용단 작업 시 불티 비산방지 조치와 화재감시자를 배치한다"
FIRE_CASE = "용접 불티가 가연물에 옮겨 붙어 화재가 발생했다"
NOISE = "잠수작업 중 공기 공급 호스가 꼬여 질식"


# ── 점수·고르기 ──


def test_score_counts_strong_twice_and_each_word_once():
    spec = SPECS["hot_ignition"]
    assert score(spec, _item(1, "불티 불티 불티", source="x")) == 2  # 같은 말은 한 번
    assert score(spec, _item(1, "불티 화재", source="x")) == 3
    assert score(spec, _item(1, "아무 관련 없는 내용", source="x")) == 0


def test_score_looks_at_the_title_too():
    spec = SPECS["hot_ignition"]
    assert score(spec, _item(1, "본문에는 없다", source="기계설비 설치 – 화재 (2021)")) == 1


def test_select_drops_items_below_min_score():
    spec = SPECS["hot_ignition"]
    pool = [_item(1, NOISE), _item(2, "산소농도를 측정한다")]
    assert select_citations(spec, pool) == []


def test_select_reserves_two_slots_for_guides_even_when_cases_score_higher():
    spec = SPECS["hot_ignition"]
    pool = [
        _item(1, "불티 가연물 화재감시자 비산방지 인화성", corpus="accident"),   # 사고사례 최고점
        _item(2, "불티 가연물 화재감시자 비산방지", corpus="accident"),          # 사고사례 2등 (전체 2등)
        _item(3, "불티 가연물 화재", corpus="guide"),                             # 조문 1등 (전체 3등)
        _item(4, "화재감시자 화재", corpus="guide"),                              # 조문 2등 (전체 4등)
    ]
    assert [i.id for i in select_citations(spec, pool)] == ["id-1", "id-3", "id-4"]  # 사고사례 1 + 조문 2, 점수순 표시


def test_select_reserved_slots_are_taken_even_if_guides_are_lower_than_cases():
    spec = SPECS["hot_ignition"]
    pool = [_item(n, "불티 가연물 화재감시자 비산방지", corpus="accident") for n in (1, 2, 3)] + [_item(4, "불티 화재", corpus="guide")]
    got = [i.id for i in select_citations(spec, pool)]
    assert "id-4" in got and len(got) == MAX_CITATIONS


def test_select_reserves_a_slot_for_a_case_even_when_guides_score_higher():
    spec = SPECS["hot_ignition"]
    pool = [_item(n, "불티 가연물 화재감시자 비산방지", corpus="guide") for n in (1, 2, 3)] + [_item(4, "불티 화재", corpus="accident")]
    assert [i.id for i in select_citations(spec, pool)] == ["id-1", "id-2", "id-4"]  # 조문 2 + 사고사례 1 (3번째 조문은 자리가 없다)


def test_select_never_exceeds_the_cap_even_if_reserved_slots_are_configured_larger(monkeypatch):
    monkeypatch.setattr(judgment_code, "MIN_PER_CORPUS", {"guide": 5, "accident": 5})
    pool = [_item(n, FIRE_GUIDE) for n in range(1, 6)] + [_item(n, FIRE_CASE, corpus="accident") for n in range(6, 10)]
    assert len(select_citations(SPECS["hot_ignition"], pool)) == MAX_CITATIONS


def test_select_without_guides_fills_with_cases():
    spec = SPECS["hot_ignition"]
    pool = [_item(n, FIRE_CASE, corpus="accident") for n in (1, 2, 3, 4)]
    assert len(select_citations(spec, pool)) == MAX_CITATIONS


def test_select_with_one_guide_and_no_cases_gives_that_guide():
    spec = SPECS["hot_ignition"]
    assert [i.id for i in select_citations(spec, [_item(1, FIRE_GUIDE)])] == ["id-1"]


def test_select_dedupes_by_id_and_caps():
    spec = SPECS["hot_ignition"]
    pool = [_item(1, FIRE_GUIDE), _item(1, FIRE_GUIDE)] + [_item(n, FIRE_CASE, corpus="accident") for n in range(2, 9)]
    got = select_citations(spec, pool)
    assert len(got) <= MAX_CITATIONS and len({i.id for i in got}) == len(got)


def test_select_keeps_pool_order_for_equal_scores():
    spec = SPECS["hot_ignition"]
    pool = [_item(n, FIRE_CASE, corpus="accident") for n in (7, 5, 6, 8)]
    assert [i.id for i in select_citations(spec, pool)] == ["id-7", "id-5", "id-6"]  # 점수가 같으면 풀에 들어온 순서


def test_every_template_has_a_spec_and_queries():
    from app.agent.judgment import _TEMPLATES

    assert set(SPECS) == set(_TEMPLATES)
    for key, spec in SPECS.items():
        assert spec.guide_query and spec.accident_query and spec.strong and spec.weak, key


def test_strong_keywords_are_not_generic_words():
    """템플릿 키워드의 흔한 말("용접", "환기")이 strong에 들어가 점수 1개로 통과하면 안 된다."""
    for key, spec in SPECS.items():
        assert "용접" not in spec.strong and "환기" not in spec.strong and "화재" not in spec.strong, key


# ── 질의 ──


def test_hazard_queries_one_per_corpus():
    h = rule_hazards(_inp())[0]
    q = hazard_queries(h, _inp())
    assert set(q) == {"guide", "accident"} and "화재감시자" in q["guide"]


def test_hazard_queries_adds_flammable_material_name():
    msds = [MsdsDto(query="톨루엔", found=True, name="톨루엔", cas_no="108-88-3", ghs_classes=["인화성 액체 2"], hazards=["H225"], source="[mock]")]
    inp = _inp(msds=msds)
    h = rule_hazards(inp)[0]
    assert "톨루엔" in hazard_queries(h, inp)["accident"]
    assert "톨루엔" not in hazard_queries(h, _inp())["accident"]


def test_hazard_queries_unknown_template_has_none():
    h = rule_hazards(_inp())[0]
    h.template = None
    assert hazard_queries(h, _inp()) == {}


# ── judge_code ──


def test_judge_code_replaces_citations_with_researched_hits():
    tools = FakeTools(guides=[_hit(1, FIRE_GUIDE, "규칙 (제241조의2)"), _hit(2, NOISE)], accidents=[_hit(3, FIRE_CASE, "기계설비 설치 – 화재")])
    old = rule_hazards(_inp())
    hazards, source, dropped = judge_code(tools, _inp())
    assert source == "fallback" and dropped == 0
    assert [h.hazard for h in hazards] == [h.hazard for h in old]  # 위험요인 목록은 템플릿 그대로
    fire = hazards[0]
    assert {c.ref_id for c in fire.citations} == {"hit-1", "hit-3"}
    assert fire.needs_review is False
    assert all(c.kind == "KB" for c in fire.citations)


def test_judge_code_only_changes_citations_and_needs_review():
    inp = _inp()
    old = rule_hazards(inp)
    hazards, _, _ = judge_code(FakeTools(guides=[_hit(1, FIRE_GUIDE)]), inp)
    for a, b in zip(old, hazards):
        assert (a.hazard, a.accident_types, a.risk_factor, a.frequency, a.severity, a.measures, a.residual_frequency, a.source, a.template) == (
            b.hazard, b.accident_types, b.risk_factor, b.frequency, b.severity, b.measures, b.residual_frequency, b.source, b.template
        )


def test_judge_code_searches_each_corpus_once_per_hazard_with_k():
    tools = FakeTools()
    hazards, _, _ = judge_code(tools, _inp())  # 화기(+용접) → hot_ignition, hot_fume
    assert [h.template for h in hazards] == ["hot_ignition", "hot_fume"]
    assert [n for n, _ in tools.calls] == ["search_guides", "search_accidents"] * 2
    assert all(a["k"] == judgment_code.PER_HAZARD_K and a["query"] for _, a in tools.calls)
    assert all("min_score" not in a for _, a in tools.calls)


def test_judge_code_does_not_repeat_identical_queries(monkeypatch):
    monkeypatch.setitem(judgment_code.SPECS, "hot_fume", judgment_code.SPECS["hot_ignition"])
    tools = FakeTools()
    judge_code(tools, _inp())
    assert len(tools.calls) == 2


def test_judge_code_uses_first_collected_evidence_too():
    # 이전 키워드(불티·가연물 …)에는 안 걸리고 새 기준(인화성)에만 걸리는 근거라서, 풀에 합치지 않으면 인용이 비어 버린다
    inp = _inp([_item(1, "인화성 물질을 취급하는 장소에서는 화기를 사용하지 않는다", source="처음 수집한 조문")])
    assert rule_hazards(inp)[0].citations == []
    hazards, _, _ = judge_code(FakeTools(), inp)  # 다시 검색은 아무것도 못 찾음
    assert [c.ref_id for c in hazards[0].citations] == ["id-1"]


def test_judge_code_keeps_template_msds_citation():
    msds = [MsdsDto(query="톨루엔", found=True, name="톨루엔", cas_no="108-88-3", ghs_classes=["인화성 액체 2"], hazards=["H225"], source="[mock]")]
    hazards, _, _ = judge_code(FakeTools(guides=[_hit(1, FIRE_GUIDE)]), _inp(msds=msds))
    kinds = [c.kind for c in hazards[0].citations]
    assert kinds.count("MSDS") == 1 and kinds[0] == "KB"


def test_judge_code_keeps_old_keyword_citations_when_nothing_passes_the_new_bar():
    """이전 키워드("용접" 하나)에는 맞지만 새 기준(2점)에는 못 미치면, 인용을 비우지 않고 이전 것을 둔다."""
    inp = _inp([_item(1, "용접 절차서를 비치한다")])
    old = rule_hazards(inp)
    assert [c.ref_id for c in old[1].citations] == ["id-1"] and old[1].template == "hot_fume"
    assert score(SPECS["hot_fume"], inp.pack.items[0]) < MIN_SCORE
    hazards, _, _ = judge_code(FakeTools(guides=[_hit(1, NOISE)]), inp)
    assert [c.ref_id for c in hazards[1].citations] == ["id-1"]
    assert hazards[1].needs_review is False


def test_judge_code_nothing_at_all_marks_needs_review():
    hazards, source, _ = judge_code(FakeTools(guides=[_hit(1, NOISE)]), _inp())
    assert source == "fallback"
    assert hazards[0].citations == [] and hazards[0].needs_review is True  # 템플릿이 정한 값 그대로


@pytest.mark.parametrize("error", [ToolError("RAG_DOWN", "검색 서버에 연결할 수 없습니다.", tool="search_guides")])
def test_judge_code_tool_error_does_not_stop_and_falls_back_to_collected(error, caplog):
    inp = _inp([_item(1, FIRE_GUIDE)])
    with caplog.at_level(logging.WARNING, logger="app.agent.judgment_code"):
        hazards, source, _ = judge_code(FakeTools(error=error), inp)
    assert source == "fallback" and [c.ref_id for c in hazards[0].citations] == ["id-1"]
    assert any("위험요인 근거 검색 실패" in r.message for r in caplog.records)


def test_judge_code_invalid_tool_output_is_skipped():
    hazards, _, _ = judge_code(FakeTools(raw={"hits": [{"id": "x"}]}), _inp([_item(1, FIRE_GUIDE)]))
    assert [c.ref_id for c in hazards[0].citations] == ["id-1"]


def test_judge_code_programming_errors_are_not_swallowed():
    with pytest.raises(RuntimeError):
        judge_code(FakeTools(error=RuntimeError("bug")), _inp())


def test_judge_code_accepts_pydantic_tool_output():
    from app.agent.evidence_collector.models import SearchToolOutput, ToolHit

    out = SearchToolOutput(hits=[ToolHit(**_hit(1, FIRE_GUIDE))])
    hazards, _, _ = judge_code(FakeTools(raw=out), _inp())
    assert "hit-1" in [c.ref_id for c in hazards[0].citations]


def test_judge_code_citation_has_kb_source_fields():
    tools = FakeTools(guides=[{**_hit(1, FIRE_GUIDE), "doc_id": "LAW:7"}])
    hazards, _, _ = judge_code(tools, _inp())
    c = hazards[0].citations[0]
    assert c.source_type == "LAW" and c.kb_collection == "kb_guides"


# ── work_register 연결 ──

_MESSAGE = "3호기 배관 용접 작업입니다. 오후 2시부터 5시까지 용접기를 쓰고 3명이 합니다. 주변 물질 없음."


class StubLLM(LLMClient):
    name = "STUB"

    def __init__(self) -> None:
        super().__init__(None)
        self.kinds: list[str] = []

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        kind = "pick" if "근거 선택 담당" in system else "judge" if "위험요인 판단 담당" in system else "plan"
        self.kinds.append(kind)
        return LLMResponse(text="" if kind != "pick" else '{"H1": ["E1"]}', stop_reason="end")


def _run(*, code: bool, monkeypatch, llm=None):
    monkeypatch.setattr(work_register, "JUDGMENT_CODE", code)
    llm = llm or StubLLM()
    agent = work_register.WorkRegisterAgent(llm=llm, tools=build_tool_registry())
    return agent.run(WorkRegisterRequest(message=_MESSAGE, run_id=1, site_id=1)), llm


def _key_paths(value, prefix=""):
    if isinstance(value, dict):
        out = set()
        for k, v in value.items():
            out |= {f"{prefix}.{k}"} | _key_paths(v, f"{prefix}.{k}")
        return out
    if isinstance(value, list):
        out = set()
        for v in value[:1]:
            out |= _key_paths(v, prefix + "[]")
        return out
    return set()


def test_default_is_code():
    assert JUDGMENT_CODE is True and work_register.JUDGMENT_CODE is True


def test_agent_code_mode_uses_no_judgment_llm_call(monkeypatch):
    result, llm = _run(code=True, monkeypatch=monkeypatch)
    assert result.run_status == "DONE"
    assert "judge" not in llm.kinds and "pick" not in llm.kinds
    step = next(s for s in result.steps if s.name == "judgment")
    assert step.output["mode"] == "code" and step.output["source"] == "fallback"
    assert result.llm_sources.judgment == "fallback"
    assert all(i.citations for i in result.drafts.risk_assessment.items)


def test_agent_code_mode_adds_only_search_tool_steps(monkeypatch):
    on, _ = _run(code=True, monkeypatch=monkeypatch)
    off, _ = _run(code=False, monkeypatch=monkeypatch)
    names_on = [s.name for s in on.steps]
    names_off = [s.name for s in off.steps]
    extra = list(names_on)
    for n in names_off:
        extra.remove(n)
    assert extra and set(extra) <= {"search_guides", "search_accidents"}
    i = names_on.index("judgment")
    assert set(names_on[i - len(extra): i]) <= {"search_guides", "search_accidents"}  # 판단 직전에 붙는다


def test_response_json_shape_is_the_same_as_before(monkeypatch):
    on, _ = _run(code=True, monkeypatch=monkeypatch)
    off, _ = _run(code=False, monkeypatch=monkeypatch)
    a = on.model_dump(by_alias=True)
    b = off.model_dump(by_alias=True)
    for d in (a, b):
        for s in d["steps"]:
            s.pop("seq", None)
    assert _key_paths(a) == _key_paths(b)
    assert a["llmSources"].keys() == b["llmSources"].keys()
    skip = ("steps", "drafts", "llmSources", "citations")  # 인용 내용은 근거 선택이 달라서 다르다 (키 구조는 위에서 비교)
    assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}


def test_agent_off_switch_goes_back_to_pick(monkeypatch):
    monkeypatch.setattr(work_register, "judge_code", lambda *a, **k: pytest.fail("끄면 judge_code를 부르면 안 됩니다"))
    result, llm = _run(code=False, monkeypatch=monkeypatch)
    assert result.run_status == "DONE"
    assert llm.kinds.count("pick") == 1
    step = next(s for s in result.steps if s.name == "judgment")
    assert step.output["mode"] == "pick"


def test_agent_code_mode_survives_search_failures(monkeypatch):
    class Boom(work_register.WorkRegisterAgent):
        pass

    registry = build_tool_registry()
    real_call = registry.call
    state = {"judging": False}

    def call(name, arguments):
        if state["judging"] and name.startswith("search_"):
            raise ToolError("RAG_DOWN", "검색 서버에 연결할 수 없습니다.", tool=name)
        return real_call(name, arguments)

    monkeypatch.setattr(work_register, "JUDGMENT_CODE", True)
    original = work_register.judge_code

    def flagged(tools, inp, **kw):
        state["judging"] = True
        try:
            return original(tools, inp, **kw)
        finally:
            state["judging"] = False

    monkeypatch.setattr(work_register, "judge_code", flagged)
    monkeypatch.setattr(registry, "call", call)
    agent = work_register.WorkRegisterAgent(llm=StubLLM(), tools=registry)
    result = agent.run(WorkRegisterRequest(message=_MESSAGE, run_id=1, site_id=1))
    assert result.run_status == "DONE"
    assert all(i.citations for i in result.drafts.risk_assessment.items)


# ── 적용 스크립트 (apply_judgment_code.py) ──


def _load_apply_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "apply_judgment_code.py"
    if not path.is_file():
        pytest.skip("apply_judgment_code.py가 저장소 루트에 없습니다")
    spec = importlib.util.spec_from_file_location("apply_judgment_code", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _picked_text(script) -> str:
    return (
        "import x\n"
        "from .judgment_pick import JUDGMENT_MODE, PICK_MAX_TOKENS, judge_pick\n"
        + script.IMPORT_OLD
        + "\ndef run():\n"
        + script.BRANCH_OLD
        + "            hazards = judge_pick()\n"
        "        else:\n"
        "            hazards = judge()\n"
        "    return hazards\n"
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
    original = _picked_text(script).replace("\n", newline).encode("utf-8")
    target.write_bytes(original)
    assert script.main([]) == 0
    patched = target.read_bytes()
    assert patched != original and b"judge_code" in patched
    assert (b"\r\n" in patched) == (newline == "\r\n")
    assert script.main([]) == 0 and target.read_bytes() == patched
    assert script.main(["--revert"]) == 0
    assert target.read_bytes() == original


def test_apply_script_stops_if_judgment_pick_is_not_applied(script_env, capsys):
    script, target = script_env
    plain = ("from .judgment import JudgeInput, judge, verify\n" + script.IMPORT_OLD).encode("utf-8")
    target.write_bytes(plain)
    assert script.main([]) == 1
    assert target.read_bytes() == plain
    assert "judgment_pick" in capsys.readouterr().out


def test_apply_script_stops_when_run_outside_repo_root(script_env):
    script, target = script_env
    assert not target.exists()
    assert script.main([]) == 2
