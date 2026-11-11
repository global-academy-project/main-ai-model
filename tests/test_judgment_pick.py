"""판단 '고르기' 방식(app/agent/judgment_pick.py) 시험. 모델은 부르지 않고 가짜 LLM을 쓴다.

검증 범위: 가짜 LLM + 전체 app 코드 실행 (실제 모델·검색 품질은 확인하지 않는다).
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("app")  # 앱 코드가 없는 환경에서는 건너뛴다

from app.agent import judgment_pick, work_register  # noqa: E402
from app.agent.evidence_collector import EvidenceItem, EvidencePack
from app.agent.judgment import JudgeInput, rule_hazards
from app.agent.judgment_pick import (
    JUDGMENT_MODE,
    PICK_MAX_CITATIONS,
    PICK_MAX_TOKENS,
    apply_picks,
    build_pick_input,
    judge_pick,
    parse_picks,
)
from app.exceptions import LLMConfigError, LLMError
from app.llm.llm_base import LLMClient
from app.llm.llm_types import LLMResponse
from app.schemas.agent_work import MsdsDto, WorkRegisterRequest
from app.tools.tool_setup import build_tool_registry


class StubLLM(LLMClient):
    name = "STUB"

    def __init__(self, picker=None, *, plan_text: str = "", judge_text: str = "", error: Exception | None = None) -> None:
        super().__init__(None)
        self.picker = picker  # 근거 선택 프롬프트에 대한 응답 (str 또는 호출 가능)
        self.plan_text = plan_text
        self.judge_text = judge_text
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        kind = "pick" if "근거 선택 담당" in system else "judge" if "위험요인 판단 담당" in system else "plan"
        self.calls.append({"kind": kind, "max_tokens": max_tokens, "user": messages[-1].content, "system": system})
        if kind == "pick":
            if self.error is not None:
                raise self.error
            text = self.picker(messages[-1].content) if callable(self.picker) else (self.picker or "")
        else:
            text = self.judge_text if kind == "judge" else self.plan_text
        return LLMResponse(text=text, stop_reason="end")

    def kinds(self) -> list[str]:
        return [c["kind"] for c in self.calls]


def _item(n: int, text: str, source: str | None = None, corpus: str = "guide") -> EvidenceItem:
    return EvidenceItem(id=f"id-{n}", doc_id="UNKNOWN", source=source or f"출처{n}", text=text, score=1.0 / n, corpus=corpus, round=1)


def _inp(items: list[EvidenceItem] | None = None, *, msds: list[MsdsDto] | None = None, work_types=("화기",), kind="용접") -> JudgeInput:
    items = items if items is not None else [
        _item(1, "제241조의2 화재감시자를 배치한다"),
        _item(2, "제629조 환기시키고 송기마스크를 지급한다"),
        _item(3, "용접 불티가 단열재에 옮겨 붙어 화재"),
        _item(4, "아세틸렌 용접장치 발생기실 출입 금지"),
    ]
    pack = EvidencePack(
        summary="요약", work_types=list(work_types), materials=[], equipment=["용접기"], plan_source="fallback",
        items=items, requirements=[], missing=[], search_failures=[], rounds=1,
    )
    return JudgeInput(
        text="3호기 배관 용접 작업", summary="요약", work_types=list(work_types), location="3호기 배관", hot_work_kind=kind,
        materials=[], equipment=["용접기"], pack=pack, msds=msds or [],
    )


def _ref_ids(hazard) -> list[str]:
    return [c.ref_id for c in hazard.citations]


# ── 프롬프트 ──


def test_build_pick_input_numbers_hazards_and_evidence():
    inp = _inp()
    hazards = rule_hazards(inp)
    user, refs = build_pick_input(inp, hazards)
    assert "[H1]" in user and "[H2]" in user and "[H3]" not in user
    assert "[E1]" in user and "[E4]" in user and "[E5]" not in user
    assert list(refs) == ["E1", "E2", "E3", "E4"]
    assert refs["E2"].ref_id == "id-2"


def test_build_pick_input_includes_found_msds_only():
    found = MsdsDto(query="아세톤", found=True, name="아세톤", ghs_classes=["인화성 액체"], hazards=["H225"])
    missing = MsdsDto(query="없는물질", found=False)
    _, refs = build_pick_input(_inp(msds=[found, missing]), rule_hazards(_inp()))
    assert "M1" in refs and "M2" not in refs


def test_snippet_is_cut_in_prompt():
    long_text = "가" * 1000
    user, _ = build_pick_input(_inp([_item(1, long_text)]), rule_hazards(_inp()))
    assert "가" * 200 not in user


# ── 응답 해석 ──


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"H1": ["E1", "E3"], "H2": []}, {1: ["E1", "E3"], 2: []}),
        ({"1": ["E2"]}, {1: ["E2"]}),
        ({"h2": "E1, e4"}, {2: ["E1", "E4"]}),  # 문자열·소문자도 받는다
        ({"H1": ["E1", "E1", "E1"]}, {1: ["E1"]}),  # 중복 제거
        ({"H3": ["E1"]}, {}),  # 범위 밖 위험요인은 버린다 (count=2)
        ({"H0": ["E1"], "memo": ["E1"], "H1": 5}, {1: []}),
    ],
)
def test_parse_picks(raw, expected):
    assert parse_picks(raw, 2) == expected


# ── apply_picks ──


def test_apply_picks_replaces_citations_and_clears_needs_review():
    inp = _inp()
    hazards = rule_hazards(inp)
    _, refs = build_pick_input(inp, hazards)
    hazards[0].needs_review = True
    changed, dropped = apply_picks(hazards, {1: ["E2", "E1"], 2: ["E4"]}, refs)
    assert (changed, dropped) == (2, 0)
    assert _ref_ids(hazards[0]) == ["id-2", "id-1"]
    assert _ref_ids(hazards[1]) == ["id-4"]
    assert hazards[0].needs_review is False
    assert [h.source for h in hazards] == ["rule", "rule"]  # 위험요인의 출처는 코드


def test_apply_picks_caps_at_limit_and_counts_unknown_refs():
    inp = _inp([_item(n, f"근거{n}") for n in range(1, 6)])
    hazards = rule_hazards(inp)
    _, refs = build_pick_input(inp, hazards)
    changed, dropped = apply_picks(hazards, {1: ["E1", "E99", "E2", "E3", "E4", "E5"]}, refs)
    assert changed == 1 and dropped == 1
    assert _ref_ids(hazards[0]) == ["id-1", "id-2", "id-3"][:PICK_MAX_CITATIONS]
    assert len(hazards[0].citations) == PICK_MAX_CITATIONS


def test_apply_picks_keeps_keyword_citations_when_nothing_valid():
    inp = _inp()
    hazards = rule_hazards(inp)
    before = [list(h.citations) for h in hazards]
    _, refs = build_pick_input(inp, hazards)
    changed, dropped = apply_picks(hazards, {1: ["E77"], 2: []}, refs)
    assert (changed, dropped) == (0, 1)
    assert [h.citations for h in hazards] == before


def test_apply_picks_keeps_template_msds_citation():
    flam = MsdsDto(query="아세톤", found=True, name="아세톤", ghs_classes=["인화성 액체"], hazards=["H225"], cas_no="67-64-1")
    inp = _inp(msds=[flam])
    hazards = rule_hazards(inp)
    assert any(c.kind == "MSDS" for c in hazards[0].citations)  # 템플릿이 붙인 MSDS 인용
    _, refs = build_pick_input(inp, hazards)
    apply_picks(hazards, {1: ["E2"]}, refs)
    kinds = [c.kind for c in hazards[0].citations]
    assert kinds[0] == "KB" and kinds.count("MSDS") == 1


# ── judge_pick ──


def test_judge_pick_success_reports_llm_and_uses_one_call():
    llm = StubLLM(json.dumps({"H1": ["E3", "E1"], "H2": ["E2"]}))
    hazards, source, dropped = judge_pick(llm, _inp())
    assert source == "llm" and dropped == 0
    assert llm.kinds() == ["pick"]
    assert llm.calls[0]["max_tokens"] == PICK_MAX_TOKENS
    assert _ref_ids(hazards[0]) == ["id-3", "id-1"]
    assert _ref_ids(hazards[1]) == ["id-2"]


def test_judge_pick_hazards_are_exactly_the_rule_templates():
    """LLM은 위험요인·점수·대책을 바꾸지 못한다."""
    inp = _inp()
    llm = StubLLM('{"H1": ["E1"], "hazards": [{"hazard": "지어낸 위험요인", "frequency": 5}]}')
    hazards, _, _ = judge_pick(llm, inp)
    expected = rule_hazards(inp)
    strip = lambda h: (h.hazard, h.accident_types, h.risk_factor, h.frequency, h.severity, h.measures, h.residual_frequency, h.residual_severity)
    assert [strip(h) for h in hazards] == [strip(h) for h in expected]


@pytest.mark.parametrize("text", ["", "죄송합니다. 모르겠습니다.", "{깨진 json", "[]", '{"unrelated": 1}'])
def test_judge_pick_unreadable_response_falls_back_without_retry(text, caplog):
    inp = _inp()
    llm = StubLLM(text)
    with caplog.at_level(logging.INFO, logger="app.agent.judgment_pick"):
        hazards, source, dropped = judge_pick(llm, inp)
    assert source == "fallback" and dropped == 0
    assert llm.kinds() == ["pick"]  # 재시도 없음
    expected = rule_hazards(inp)
    assert [(h.hazard, _ref_ids(h)) for h in hazards] == [(h.hazard, _ref_ids(h)) for h in expected]
    assert any("키워드 인용으로 계속" in r.getMessage() for r in caplog.records)


def test_judge_pick_logs_head_of_unreadable_response(caplog):
    with caplog.at_level(logging.WARNING, logger="app.agent.judgment_pick"):
        judge_pick(StubLLM("엉뚱한 답변 " + "x" * 500), _inp())
    message = " ".join(r.getMessage() for r in caplog.records)
    assert "엉뚱한 답변" in message and "x" * 300 not in message


def test_judge_pick_llm_error_does_not_stop_the_run():
    for error in (LLMError("timeout"), LLMConfigError("no model")):
        hazards, source, dropped = judge_pick(StubLLM(error=error), _inp())
        assert source == "fallback" and dropped == 0 and hazards


def test_judge_pick_non_llm_errors_are_not_swallowed():
    with pytest.raises(RuntimeError):
        judge_pick(StubLLM(error=RuntimeError("bug")), _inp())


def test_judge_pick_unknown_refs_are_counted_and_all_unknown_is_fallback():
    hazards, source, dropped = judge_pick(StubLLM('{"H1": ["E50"], "H2": ["E51", "E52"]}'), _inp())
    assert (source, dropped) == ("fallback", 3)


def test_judge_pick_partial_unknown_refs_still_llm():
    hazards, source, dropped = judge_pick(StubLLM('{"H1": ["E1", "E50"]}'), _inp())
    assert (source, dropped) == ("llm", 1)
    assert _ref_ids(hazards[0]) == ["id-1"]


def test_judge_pick_works_with_no_evidence():
    hazards, source, _ = judge_pick(StubLLM('{"H1": ["E1"]}'), _inp([]))
    assert source == "fallback" and hazards


# ── work_register 연결 ──

_MESSAGE = "3호기 배관 용접 작업입니다. 오후 2시부터 5시까지 용접기를 쓰고 3명이 합니다. 주변 물질 없음."


def _run(llm, *, mode: str, monkeypatch):
    monkeypatch.setattr(work_register, "JUDGMENT_MODE", mode)
    monkeypatch.setattr(work_register, "JUDGMENT_CODE", False, raising=False)  # judgment_code 패치가 있어도 pick/full 경로를 시험한다
    agent = work_register.WorkRegisterAgent(llm=llm, tools=build_tool_registry())
    return agent.run(WorkRegisterRequest(message=_MESSAGE, run_id=1, site_id=1))


def _phase(result, name: str):
    return next(s for s in result.steps if s.name == name)


def test_default_mode_is_pick():
    assert JUDGMENT_MODE == "pick" and work_register.JUDGMENT_MODE == "pick"


def test_agent_pick_mode_end_to_end(monkeypatch):
    llm = StubLLM(lambda user: '{"H1": ["E1"], "H2": ["E1", "E2"]}')
    result = _run(llm, mode="pick", monkeypatch=monkeypatch)
    assert result.run_status == "DONE"
    assert "judge" not in llm.kinds() and llm.kinds().count("pick") == 1
    assert [c["max_tokens"] for c in llm.calls if c["kind"] == "pick"] == [PICK_MAX_TOKENS]
    assert result.llm_sources.judgment == "llm"
    step = _phase(result, "judgment")
    assert step.output["mode"] == "pick" and step.output["source"] == "llm"
    items = result.drafts.risk_assessment.items
    assert [i.source for i in items] == ["rule", "rule"]
    assert all(i.citations for i in items)


def test_agent_pick_mode_llm_garbage_still_completes(monkeypatch):
    llm = StubLLM("모름")
    result = _run(llm, mode="pick", monkeypatch=monkeypatch)
    assert result.run_status == "DONE"
    assert result.llm_sources.judgment == "fallback"
    assert llm.kinds().count("pick") == 1  # 재시도 없음


def test_agent_full_mode_is_the_old_behavior(monkeypatch):
    """끄면(full) 이전 방식: judge()를 쓰고, 단계 출력에 mode 키가 없다."""
    calls = []
    original = work_register.judge

    def spy(llm, inp, *, max_tokens):
        calls.append(max_tokens)
        return original(llm, inp, max_tokens=max_tokens)

    monkeypatch.setattr(work_register, "judge", spy)
    monkeypatch.setattr(judgment_pick, "judge_pick", lambda *a, **k: pytest.fail("full 모드에서 judge_pick을 부르면 안 됩니다"))
    llm = StubLLM(judge_text="읽을 수 없음")
    result = _run(llm, mode="full", monkeypatch=monkeypatch)
    assert result.run_status == "DONE"
    assert calls == [1200]  # 이전 기본값 그대로
    assert "pick" not in llm.kinds() and llm.kinds().count("judge") == 2  # 이전처럼 한 번 더 시도
    assert set(_phase(result, "judgment").output) == {"source", "hazards"}
    assert result.llm_sources.judgment == "fallback"


# ── 적용 스크립트 (apply_judgment_pick.py) ──


def _load_apply_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "apply_judgment_pick.py"
    if not path.is_file():
        pytest.skip("apply_judgment_pick.py가 저장소 루트에 없습니다")
    spec = importlib.util.spec_from_file_location("apply_judgment_pick", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unpatched_text(script) -> str:
    return (
        "import x\n"
        + script.IMPORT_OLD
        + "from .slots import y\n\n"
        + "def run():\n"
        + script.CALL_OLD
        + "    return hazards\n"
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
    assert patched != original and b"judge_pick" in patched
    assert (b"\r\n" in patched) == (newline == "\r\n")  # 줄바꿈 유지
    added = len(script.IMPORT_NEW.splitlines()) - len(script.IMPORT_OLD.splitlines()) + len(script.CALL_NEW.splitlines()) - len(script.CALL_OLD.splitlines())
    assert patched.replace(b"\r\n", b"\n").count(b"\n") == original.replace(b"\r\n", b"\n").count(b"\n") + added
    assert script.main([]) == 0 and target.read_bytes() == patched  # 두 번째는 바꾸지 않는다
    assert script.main(["--revert"]) == 0
    assert target.read_bytes() == original


def test_apply_script_stops_without_changing_when_anchor_missing(script_env):
    script, target = script_env
    other = ("from .judgment import JudgeInput, judge, verify\n" + "x = 1\n").encode("utf-8")
    target.write_bytes(other)
    assert script.main([]) == 1
    assert target.read_bytes() == other


def test_apply_script_stops_when_run_outside_repo_root(script_env):
    script, target = script_env  # 파일을 만들지 않았다
    assert not target.exists()
    assert script.main([]) == 2
