"""계획 단계 LoRA 어댑터 연결 (plan_adapter.py, llm_work_register.py, plan_brief.BriefPlanner).

기본(LLM_ADAPTERS 에 work_register 없음)은 변경 전과 같고, 어댑터를 쓸 수 있을 때만 학습 때와 같은 짧은 프롬프트가 나간다.
모델은 부르지 않는다 (가짜 클라이언트).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.agent import plan_adapter, plan_brief
from app.agent.evidence_collector import WorkContext
from app.agent.evidence_collector.planner import build_planner_input
from app.agent.evidence_collector.vocab import DEFAULT_CORPORA
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, adapter_ready, build_adapter_input
from app.agent.plan_brief import BRIEF_MAX_TOKENS, BriefPlanner, build_brief_prompt
from app.llm import adapter_paths, llm_transformers
from app.llm.engines import llm_small_site_guide, llm_work_register
from app.llm.engines.llm_work_register import ADAPTER_NAME, WorkRegisterClient, build_work_register_client
from app.llm.llm_fake import FakeLLMClient, text_response

TEXT = "내일 3호기 펌프 베어링 교체. 고착되면 토치로 가열. 옆에 메탄올 드럼 있음. 호이스트는 안 씀."
GOOD = '{"summary": "3호기 펌프 베어링 교체, 고착 시 토치 가열", "equipment": ["토치"], "materials": ["메탄올"]}'


@pytest.fixture(autouse=True)
def _no_builtin_adapter(monkeypatch):
    """이 파일의 시험은 PC·서버에 실제 어댑터 폴더(adapters/work_register)가 있든 없든 같아야 한다. 내장 위치 시험은 test_adapter_builtin.py."""
    monkeypatch.setattr(adapter_paths, "BUILTIN_ADAPTERS", {})


def make_work(**kw) -> WorkContext:
    base = dict(text=TEXT, location="3호기 펌프", hot_work_kind="토치", nearby_materials=["메탄올"], headcount=2,
                end_time="17:00", equipment=["토치"], work_types=["화기"])
    base.update(kw)
    return WorkContext(**base)


@pytest.fixture
def adapter_dir(tmp_path, monkeypatch):
    """LLM_ADAPTERS 에 work_register 가 있고 폴더에 adapter_config.json 이 있는 상태."""
    folder = tmp_path / "plan_brief"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={ADAPTER_NAME: str(folder)}))
    return folder


def llm_with(adapter: str | None, *responses: str) -> FakeLLMClient:
    llm = FakeLLMClient([text_response(r) for r in responses])
    if adapter is not None:
        llm.adapter = adapter
    return llm


# ── 짧은 프롬프트의 원본 ──


def test_system_prompt_is_the_agreed_one_liner():
    assert ADAPTER_SYSTEM_PROMPT == "작업 내용에서 summary, equipment, materials를 JSON 객체 하나로 뽑으세요."
    assert "\n" not in ADAPTER_SYSTEM_PROMPT


def test_adapter_input_has_text_materials_and_equipment_only():
    assert build_adapter_input(make_work()) == (
        "작업 내용: 내일 3호기 펌프 베어링 교체. 고착되면 토치로 가열. 옆에 메탄올 드럼 있음. 호이스트는 안 씀.\n"
        "주변 물질: 메탄올\n"
        "투입 장비: 토치"
    )


def test_adapter_input_leaves_out_empty_slots_and_strips_the_text():
    work = make_work(text="  용접 작업  \n", nearby_materials=[], equipment=[])
    assert build_adapter_input(work) == "작업 내용: 용접 작업"
    assert build_adapter_input(make_work(nearby_materials=["톨루엔", "아세톤"], equipment=[])) == (
        f"작업 내용: {TEXT}\n주변 물질: 톨루엔, 아세톤"
    )


def test_adapter_input_never_carries_the_other_slots():
    out = build_adapter_input(make_work())
    for label in ("장소", "화기 종류", "인원", "종료 시간", "3호기 펌프\n", "17:00", "2명"):
        assert label not in out.split("작업 내용:")[1].split("\n", 1)[-1]


def test_adapter_input_is_much_shorter_than_the_default_one():
    assert len(build_adapter_input(make_work())) < len(build_planner_input(make_work()))
    assert len(ADAPTER_SYSTEM_PROMPT) * 5 < len(build_brief_prompt())


# ── 어댑터를 쓸 수 있나 ──


def test_not_ready_without_an_adapter_name_on_the_client(adapter_dir):
    assert adapter_ready(llm_with(None)) is False
    assert adapter_ready(llm_with("extract")) is False  # 다른 어댑터 이름은 계획용이 아니다


def test_another_adapter_with_a_real_folder_is_still_not_the_plan_adapter(adapter_dir, monkeypatch):
    settings = SimpleNamespace(llm_adapters={"extract": str(adapter_dir), ADAPTER_NAME: str(adapter_dir)})
    monkeypatch.setattr("app.core.get_settings", lambda: settings)
    assert adapter_ready(llm_with("extract")) is False
    assert adapter_ready(llm_with(ADAPTER_NAME)) is True


def test_not_ready_when_the_name_is_not_configured(monkeypatch):
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={}))
    assert adapter_ready(llm_with(ADAPTER_NAME)) is False
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters=None))
    assert adapter_ready(llm_with(ADAPTER_NAME)) is False


def test_not_ready_when_the_folder_or_its_config_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={ADAPTER_NAME: str(tmp_path / "nope")}))
    assert adapter_ready(llm_with(ADAPTER_NAME)) is False
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={ADAPTER_NAME: str(tmp_path / "empty")}))
    assert adapter_ready(llm_with(ADAPTER_NAME)) is False  # adapter_config.json 이 없다


def test_not_ready_when_settings_cannot_be_read(monkeypatch):
    def boom():
        raise RuntimeError("설정 오류")

    monkeypatch.setattr("app.core.get_settings", boom)
    assert adapter_ready(llm_with(ADAPTER_NAME)) is False


def test_ready_when_named_configured_and_folder_has_the_config(adapter_dir):
    assert adapter_ready(llm_with(ADAPTER_NAME)) is True


def test_not_ready_after_the_engine_gave_up_attaching(adapter_dir, monkeypatch, caplog):
    llm = llm_with(ADAPTER_NAME)
    llm.config = "cfg"
    monkeypatch.setattr(llm_transformers, "_cache", {"cfg": SimpleNamespace(skipped={ADAPTER_NAME}, adapters=set())})
    with caplog.at_level(logging.WARNING, logger="app.agent.plan_adapter"):
        assert adapter_ready(llm) is False
    assert "긴 프롬프트" in caplog.text
    # 붙이기에 성공한 모델이면 그대로 쓴다
    monkeypatch.setattr(llm_transformers, "_cache", {"cfg": SimpleNamespace(skipped=set(), adapters={ADAPTER_NAME})})
    assert adapter_ready(llm) is True
    # 모델을 아직 읽기 전(캐시에 없음)이면 실패한 것이 아니다
    monkeypatch.setattr(llm_transformers, "_cache", {})
    assert adapter_ready(llm) is True


def test_unreadable_model_cache_does_not_block_the_adapter(adapter_dir, monkeypatch):
    llm = llm_with(ADAPTER_NAME)
    llm.config = "cfg"
    monkeypatch.setattr(llm_transformers, "_cache", None)
    assert adapter_ready(llm) is True


# ── 엔진: LLM_ADAPTERS 에 이름이 있을 때만 어댑터 이름을 붙인다 ──


def settings_with(adapters):
    return SimpleNamespace(ai_max_output_tokens=800, llm_adapters=adapters)


def test_engine_adapter_defaults_to_none():
    assert WorkRegisterClient.adapter is None
    assert ADAPTER_NAME == "work_register"
    assert llm_work_register.ADAPTER_ENABLED is True


def test_engine_has_no_adapter_unless_configured():
    assert build_work_register_client(settings=settings_with({})).adapter is None
    assert build_work_register_client(settings=SimpleNamespace(ai_max_output_tokens=800)).adapter is None  # llm_adapters 없는 설정
    assert build_work_register_client(settings=settings_with(None)).adapter is None
    assert build_work_register_client(settings=settings_with({"extract": "/x"})).adapter is None


def test_engine_gets_the_name_when_configured():
    client = build_work_register_client(settings=settings_with({ADAPTER_NAME: "/models/adapters/plan_brief"}))
    assert client.adapter == ADAPTER_NAME
    assert WorkRegisterClient.adapter is None  # 이 인스턴스만 바뀐다


def test_engine_switch_off_wins_over_configuration(monkeypatch):
    monkeypatch.setattr(llm_work_register, "ADAPTER_ENABLED", False)
    assert build_work_register_client(settings=settings_with({ADAPTER_NAME: "/x"})).adapter is None


def test_other_engines_are_not_given_the_plan_adapter():
    client = llm_small_site_guide.build_small_site_guide_client(settings=settings_with({ADAPTER_NAME: "/x"}))
    assert client.adapter is None


def test_engine_model_config_is_unchanged_by_the_adapter():
    plain = build_work_register_client(settings=settings_with({}))
    tuned = build_work_register_client(settings=settings_with({ADAPTER_NAME: "/x"}))
    assert plain.config == tuned.config  # 같은 모델 설정 → 같은 기본 모델 한 벌을 공유한다


# ── BriefPlanner: 어떤 프롬프트가 나가나 ──


def plan_with(llm: FakeLLMClient, work: WorkContext | None = None):
    planner = BriefPlanner(llm=llm, corpora=DEFAULT_CORPORA)
    return planner, planner.plan(work or make_work())


def test_default_prompt_is_exactly_the_old_one():
    llm = llm_with(None, GOOD)
    planner, _ = plan_with(llm)
    call = llm.calls[0]
    assert call["system"] == build_brief_prompt()
    assert call["messages"][0].content == build_planner_input(make_work())
    assert call["max_tokens"] == BRIEF_MAX_TOKENS
    assert planner._short is False


def test_adapter_prompt_is_the_short_one(adapter_dir):
    llm = llm_with(ADAPTER_NAME, GOOD)
    planner, _ = plan_with(llm)
    call = llm.calls[0]
    assert call["system"] == ADAPTER_SYSTEM_PROMPT
    assert call["messages"][0].role == "user"
    assert call["messages"][0].content == build_adapter_input(make_work())
    assert call["max_tokens"] == BRIEF_MAX_TOKENS
    assert planner._short is True


def test_adapter_name_without_a_usable_folder_keeps_the_long_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={ADAPTER_NAME: str(tmp_path / "missing")}))
    llm = llm_with(ADAPTER_NAME, GOOD)
    plan_with(llm)
    assert llm.calls[0]["system"] == build_brief_prompt()


def test_both_prompts_give_the_same_validated_plan(adapter_dir):
    _, long_plan = plan_with(llm_with(None, GOOD))
    _, short_plan = plan_with(llm_with(ADAPTER_NAME, GOOD))
    assert long_plan == short_plan
    assert short_plan.summary == "3호기 펌프 베어링 교체, 고착 시 토치 가열"
    assert short_plan.equipment == ["토치"] and short_plan.materials == ["메탄올"]
    assert short_plan.queries == [] and short_plan.source == "llm"


def test_candidate_validation_still_applies_to_the_adapter_output(adapter_dir):
    bad = '{"summary": "베어링 교체 99명", "equipment": ["호이스트", "토치"], "materials": ["염산", "메탄올"]}'
    _, plan = plan_with(llm_with(ADAPTER_NAME, bad))
    assert plan.equipment == ["토치"]  # 호이스트는 부정됨(원문: 호이스트는 안 씀)
    assert plan.materials == ["메탄올"]  # 염산은 원문에 없음
    assert plan.source == "fallback"  # 원문에 없는 숫자(99)가 든 요약은 쓰지 않는다


def test_retry_keeps_the_short_input_and_adds_the_suffix(adapter_dir):
    llm = llm_with(ADAPTER_NAME, "JSON이 아님", GOOD)
    plan_with(llm)
    assert len(llm.calls) == 2
    assert llm.calls[1]["system"] == ADAPTER_SYSTEM_PROMPT
    assert llm.calls[1]["messages"][0].content == build_adapter_input(make_work()) + plan_brief._RETRY_SUFFIX


def test_two_bad_answers_fall_back_to_the_rules(adapter_dir):
    llm = llm_with(ADAPTER_NAME, "x", "y")
    _, plan = plan_with(llm)
    assert plan.queries == [] and plan.source == "fallback"
    assert len(llm.calls) == 2


def test_collector_picks_the_prompt_from_its_own_client(adapter_dir):
    """같은 설정이어도 어댑터 이름이 없는 클라이언트(예: F)는 긴 프롬프트를 쓴다."""
    with_adapter = plan_brief.StagedCollector(llm=llm_with(ADAPTER_NAME), tools=SimpleNamespace(), brief=True)
    without = plan_brief.StagedCollector(llm=llm_with(None), tools=SimpleNamespace(), brief=True)
    assert with_adapter._planner._short is True
    assert without._planner._short is False


def test_non_brief_collector_is_untouched(adapter_dir):
    collector = plan_brief.StagedCollector(llm=llm_with(ADAPTER_NAME), tools=SimpleNamespace(), brief=False)
    assert not isinstance(collector._planner, BriefPlanner)


def test_modules_import_without_a_cycle():
    for module in ("app.agent.plan_adapter", "app.agent.plan_brief", "app.llm.engines.llm_work_register", "app.agent.work_register"):
        out = subprocess.run([sys.executable, "-B", "-c", f"import {module}"], capture_output=True, text=True)
        assert out.returncode == 0, (module, out.stderr[-400:])
