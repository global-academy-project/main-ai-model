"""어댑터 내장 위치 (app/llm/adapter_paths.py, 11/18): 환경변수 없이 adapters/work_register 에 어댑터 파일이 있는지로만 정한다.

파일이 있으면 어댑터와 짧은 프롬프트, 없으면 어댑터 없이 긴 프롬프트 (내장 위치를 만들기 전과 같다). 이 이름은 LLM_ADAPTERS 를 보지 않는다.
모델은 부르지 않는다 (가짜 클라이언트·가짜 peft). 실제 어댑터 폴더가 이 PC 에 있든 없든 같은 결과여야 해서, 내장 위치는 항상 임시 폴더로 바꿔 시험한다.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent.evidence_collector import vocab
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, adapter_ready
from app.agent.plan_brief import BriefPlanner, build_brief_prompt
from app.llm import adapter_paths, llm_transformers
from app.llm.adapter_paths import BUILTIN_ADAPTERS as REAL_BUILTIN
from app.llm.adapter_paths import adapter_folders
from app.llm.engines import llm_small_site_guide, llm_work_register
from app.llm.engines.llm_work_register import ADAPTER_NAME, WorkRegisterClient, build_work_register_client
from app.llm.llm_base import LLMClient
from app.llm.llm_fake import FakeLLMClient, text_response
from app.llm.llm_transformers import TransformersConfig
from app.llm.llm_types import Message
from tests.test_llm_adapters import _FakePeft
from tests.test_plan_adapter import GOOD, make_work
from tests.test_scenario_engines import _FakeTransformers

ROOT = Path(__file__).resolve().parents[1]
ENGINE_LOG = "app.llm.engines.llm_work_register"


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """내장 위치에 어댑터 폴더(adapter_config.json 있음)가 있고, 환경변수(LLM_ADAPTERS)는 비어 있다."""
    path = tmp_path / "adapters" / "work_register"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(adapter_paths, "BUILTIN_ADAPTERS", {ADAPTER_NAME: path})
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={}))
    return path


@pytest.fixture
def no_folder(tmp_path, monkeypatch):
    """내장 위치에 폴더가 없고, 환경변수도 비어 있다."""
    monkeypatch.setattr(adapter_paths, "BUILTIN_ADAPTERS", {ADAPTER_NAME: tmp_path / "adapters" / "work_register"})
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={}))


def engine_settings(**kw):
    return SimpleNamespace(ai_max_output_tokens=800, **kw)


# ── 내장 위치 ──


def test_builtin_location_is_adapters_plan_brief_under_the_repo_root():
    assert (ROOT / "app").is_dir()
    assert dict(REAL_BUILTIN) == {ADAPTER_NAME: ROOT / "adapters" / "work_register"}
    assert ADAPTER_NAME == "plan_brief"


# ── adapter_folders ──


@pytest.mark.parametrize("settings", [SimpleNamespace(llm_adapters={}), SimpleNamespace(llm_adapters=None), SimpleNamespace()])
def test_folder_is_found_without_any_environment_setting(folder, settings):
    assert adapter_folders(settings) == {ADAPTER_NAME: str(folder)}


def test_both_adapter_files_are_needed(tmp_path, monkeypatch):
    path = tmp_path / "work_register"
    monkeypatch.setattr(adapter_paths, "BUILTIN_ADAPTERS", {ADAPTER_NAME: path})
    settings = SimpleNamespace(llm_adapters={})
    assert adapter_folders(settings) == {}  # 폴더 없음
    path.mkdir()
    assert adapter_folders(settings) == {}  # 빈 폴더
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_folders(settings) == {}  # 설정만 있고 가중치 없음
    (path / "adapter_model.safetensors").write_bytes(b"x")
    assert adapter_folders(settings) == {ADAPTER_NAME: str(path)}
    (path / "adapter_config.json").unlink()
    assert adapter_folders(settings) == {}  # 가중치만 있고 설정 없음
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").unlink()
    (path / "adapter_model.bin").write_bytes(b"x")
    assert adapter_folders(settings) == {ADAPTER_NAME: str(path)}  # .bin 도 가중치로 본다


def test_llm_adapters_is_ignored_for_the_builtin_name(folder, tmp_path):
    other = str(tmp_path / "elsewhere")
    assert adapter_folders(SimpleNamespace(llm_adapters={ADAPTER_NAME: other})) == {ADAPTER_NAME: str(folder)}  # 환경변수 경로가 아니라 내장 폴더


def test_llm_adapters_cannot_switch_the_builtin_name_on(no_folder):
    assert adapter_folders(SimpleNamespace(llm_adapters={ADAPTER_NAME: "/models/adapters/plan_brief"})) == {}  # 파일이 없으면 환경변수가 있어도 없는 것


def test_other_adapters_are_kept_and_the_settings_are_not_changed(folder):
    configured = {"extract": "/models/adapters/extract"}
    result = adapter_folders(SimpleNamespace(llm_adapters=configured))
    assert result == {"extract": "/models/adapters/extract", ADAPTER_NAME: str(folder)}
    assert configured == {"extract": "/models/adapters/extract"}  # 설정 dict 를 건드리지 않는다
    assert all(isinstance(v, str) for v in result.values())
    assert adapter_folders(SimpleNamespace(llm_adapters={"extract": "/x"})) == {"extract": "/x", ADAPTER_NAME: str(folder)}  # 호출 뒤에도 같다


def test_emptied_builtin_table_changes_nothing(monkeypatch):
    monkeypatch.setattr(adapter_paths, "BUILTIN_ADAPTERS", {})
    assert adapter_folders(SimpleNamespace(llm_adapters={"extract": "/x"})) == {"extract": "/x"}


# ── 엔진: 폴더가 있으면 이름을 붙인다 ──


def test_engine_attaches_the_adapter_when_the_folder_exists(folder, caplog):
    with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
        client = build_work_register_client(settings=engine_settings())  # llm_adapters 설정 자체가 없다
    assert client.adapter == ADAPTER_NAME
    assert WorkRegisterClient.adapter is None  # 이 인스턴스만 바뀐다
    assert any("계획 어댑터" in r.message and str(folder) in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_engine_without_a_folder_goes_on_without_the_adapter_and_warns(no_folder, caplog):
    with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
        client = build_work_register_client(settings=engine_settings(llm_adapters={}))
    assert client.adapter is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "긴 프롬프트" in warnings[0] and "adapters" in warnings[0] and "work_register" in warnings[0]


def test_engine_switch_off_wins_and_stays_quiet(folder, monkeypatch, caplog):
    monkeypatch.setattr(llm_work_register, "ADAPTER_ENABLED", False)
    with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
        assert build_work_register_client(settings=engine_settings()).adapter is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_engine_ignores_llm_adapters_when_the_adapter_files_are_missing(no_folder, caplog):
    with caplog.at_level(logging.INFO, logger=ENGINE_LOG):
        client = build_work_register_client(settings=engine_settings(llm_adapters={ADAPTER_NAME: "/models/adapters/plan_brief"}))
    assert client.adapter is None  # 환경변수가 있어도 파일이 없으면 어댑터 없이
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_the_shared_model_config_and_other_engines_are_untouched(folder):
    tuned = build_work_register_client(settings=engine_settings())
    plain = build_work_register_client(settings=engine_settings(), model_id=None)
    assert tuned.config == plain.config  # 같은 모델 설정 → 같은 기본 모델 한 벌
    assert llm_small_site_guide.build_small_site_guide_client(settings=engine_settings()).adapter is None


# ── 짧은 프롬프트 판정과 계획 프롬프트 ──


def llm_like_the_engine(settings) -> FakeLLMClient:
    """엔진이 정한 어댑터 이름을 그대로 가진 가짜 클라이언트 (엔진 → 계획기 연결을 모델 없이 본다)."""
    llm = FakeLLMClient([text_response(GOOD)])
    llm.adapter = build_work_register_client(settings=settings).adapter
    return llm


def test_adapter_ready_follows_the_folder(folder, tmp_path):
    client = SimpleNamespace(adapter=ADAPTER_NAME)
    assert adapter_ready(client) is True  # 환경변수 없음
    assert adapter_ready(SimpleNamespace(adapter="extract")) is False  # 다른 이름은 계획용이 아니다
    assert adapter_ready(SimpleNamespace(adapter=None)) is False
    (folder / "adapter_model.safetensors").unlink()
    assert adapter_ready(client) is False  # 가중치가 사라지면 긴 프롬프트로


def test_adapter_ready_is_false_without_a_folder(no_folder, monkeypatch):
    assert adapter_ready(SimpleNamespace(adapter=ADAPTER_NAME)) is False
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={ADAPTER_NAME: "/models/adapters/plan_brief"}))
    assert adapter_ready(SimpleNamespace(adapter=ADAPTER_NAME)) is False  # 환경변수가 있어도 같다


def test_planner_uses_the_short_prompt_when_the_folder_exists(folder):
    llm = llm_like_the_engine(engine_settings())
    assert llm.adapter == ADAPTER_NAME
    plan = BriefPlanner(llm=llm, corpora=vocab.DEFAULT_CORPORA).plan(make_work())
    assert llm.calls[0]["system"] == ADAPTER_SYSTEM_PROMPT
    assert plan.source == "llm"


def test_planner_keeps_the_old_long_prompt_without_a_folder(no_folder):
    llm = llm_like_the_engine(engine_settings())
    assert llm.adapter is None
    plan = BriefPlanner(llm=llm, corpora=vocab.DEFAULT_CORPORA).plan(make_work())
    assert llm.calls[0]["system"] == build_brief_prompt()  # 변경 전과 같다
    assert plan.source == "llm"


# ── 붙이는 쪽 (llm_transformers) ──


def test_transformers_folder_table_includes_the_builtin_folder(folder, monkeypatch):
    assert llm_transformers._adapter_paths() == {ADAPTER_NAME: str(folder)}
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={"extract": "/x", ADAPTER_NAME: "/y"}))
    assert llm_transformers._adapter_paths() == {"extract": "/x", ADAPTER_NAME: str(folder)}


def test_transformers_folder_table_is_unchanged_without_a_folder(no_folder, monkeypatch):
    assert llm_transformers._adapter_paths() == {}
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_adapters={"extract": "/x"}))
    assert llm_transformers._adapter_paths() == {"extract": "/x"}


def test_unreadable_settings_still_means_no_adapters(folder, monkeypatch):
    def boom():
        raise RuntimeError("설정 오류")

    monkeypatch.setattr("app.core.get_settings", boom)
    assert llm_transformers._adapter_paths() == {}  # 지금처럼: 설정을 못 읽으면 어댑터 없이


@pytest.fixture
def fake_peft(monkeypatch):
    tf = _FakeTransformers()
    monkeypatch.setitem(sys.modules, "transformers", tf.module)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=_FakePeft))
    _FakePeft.calls = []
    llm_transformers.clear_model_cache()
    yield
    llm_transformers.clear_model_cache()


def _chat(adapter: str | None) -> str:
    client = LLMClient(TransformersConfig(model_id="Qwen/Qwen3-0.6B", max_new_tokens=32))
    client.adapter = adapter
    return client.chat([Message(role="user", content="내일 작업")], system="JSON만").text


def test_the_builtin_folder_is_what_gets_attached(folder, fake_peft):
    assert _chat(ADAPTER_NAME) == "모델 응답"
    assert _FakePeft.calls == [("from_pretrained", str(folder), ADAPTER_NAME)]


def test_without_a_folder_the_base_model_answers_and_peft_is_not_used(no_folder, fake_peft):
    assert _chat(ADAPTER_NAME) == "모델 응답"
    assert _FakePeft.calls == []
