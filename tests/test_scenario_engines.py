"""시나리오 엔진 모듈 테스트 (엔진 이름 = 시나리오 키, app/llm/engines/llm_<키>.py).

torch·transformers 없이 돈다: 테스트 안에서 가짜 `transformers` 모듈을 끼워 llm_categorizator의
모델 읽기 → 손질 지점 → 생성 → 출력 흐름을 그대로 지나간다.
각 팀원이 자기 모듈의 SPEC을 바꿔도 이 테스트는 깨지지 않게, 모듈 값 대신 규칙만 확인한다.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any

import pytest

from app.core import Settings
from app.core.scenario_keys import MODEL_KEYS, SCENARIO_NAMES
from app.exceptions import LLMConfigError, LLMError
from app.llm import llm_transformers
from app.llm.engines import ENGINE_MODULES, SCENARIO_ENGINE_NAMES, build_scenario_engine, engine_module
from app.llm.engines._base import HOOK_NAMES, EngineSpec, resolve_config
from app.llm.engines.llm_categorizator import CategorizatorClient
from app.llm.llm_base import LLMClient
from app.llm.llm_factory import build_llm_client
from app.llm.llm_provider import EngineBinding, LLMClientProvider, parse_bindings
from app.llm.llm_types import Message

ENGINE_KEYS = tuple(k for k in MODEL_KEYS if k != "default")  # default는 엔진이 아니라 기본 설정
CAMEL = {"router": "Router", "work_register": "WorkRegister", "confined_space_gate": "ConfinedSpaceGate",
         "near_miss_warning": "NearMissWarning", "daily_safety_log": "DailySafetyLog", "audit_query": "AuditQuery",
         "small_site_guide": "SmallSiteGuide"}
OWNERS = {"router": "B", "work_register": "B", "confined_space_gate": "C", "near_miss_warning": "A",
          "daily_safety_log": "D", "audit_query": "D", "small_site_guide": "B", "default": "공용(B)"}


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {"network_env": "OPEN", "check_rules_tool": "MOCK"}
    base.update(over)
    return Settings(**base)


# ── 가짜 transformers ──

class _Ids(list):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __getitem__(self, item: Any) -> Any:
        got = super().__getitem__(item)
        return _Ids(got) if isinstance(item, slice) else got


class _Batch(dict):
    def to(self, device: str) -> "_Batch":
        return self


class _FakeTransformers:
    def __init__(self) -> None:
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.reply = " 모델 응답 "
        self.fail_load = False
        outer = self

        class AutoTokenizer:
            eos_token_id = 0

            @classmethod
            def from_pretrained(cls, model_id: str, **kw: Any) -> "AutoTokenizer":
                return cls()

            def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> _Batch:
                outer.template_calls.append((messages, kw))
                return _Batch(input_ids=_Ids([1, 2, 3]))

            def decode(self, ids: Any, skip_special_tokens: bool) -> str:
                return outer.reply

        class AutoModelForCausalLM:
            device = "cpu"

            @classmethod
            def from_pretrained(cls, model_id: str, **kw: Any) -> "AutoModelForCausalLM":
                if outer.fail_load:
                    raise OSError("없는 모델 (원문은 로그에만)")
                outer.loads.append((model_id, kw))
                return cls()

            def to(self, device: str) -> "AutoModelForCausalLM":
                return self

            def eval(self) -> None:
                pass

            def generate(self, **kw: Any) -> list[_Ids]:
                outer.generate_calls.append(kw)
                return [_Ids([*kw["input_ids"], 7, 0])]  # 새 토큰 2개, 마지막이 eos

        self.module = types.SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM)


@pytest.fixture
def fake_tf(monkeypatch: pytest.MonkeyPatch) -> _FakeTransformers:
    fake = _FakeTransformers()
    monkeypatch.setitem(sys.modules, "transformers", fake.module)
    llm_transformers.clear_model_cache()
    yield fake
    llm_transformers.clear_model_cache()


def _say(engine: LLMClient, text: str = "분류해줘", system: str = "JSON만") -> Any:
    return engine.chat([Message(role="user", content=text)], system=system, max_tokens=64)


# ── 등록표·모듈 규칙 ──

def _client_class(name: str) -> type:
    return getattr(engine_module(name), f"{CAMEL[name]}Client")


def test_one_module_per_scenario_key() -> None:
    assert set(ENGINE_MODULES) == set(MODEL_KEYS) == set(OWNERS)
    assert SCENARIO_ENGINE_NAMES == ENGINE_KEYS and "default" not in SCENARIO_ENGINE_NAMES
    assert {SCENARIO_NAMES[c] for c in "ABCDEF"} <= set(ENGINE_MODULES)


@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_module_shape(name: str) -> None:
    module = engine_module(name)
    assert module.__name__ == f"app.llm.engines.llm_{name}" and module.OWNER == OWNERS[name]
    assert module.ENGINE_NAME == name.upper()  # config.py의 Engine 이름 (WORK_REGISTER 등)
    assert isinstance(module.SPEC, EngineSpec)
    own = {n: v for n, v in vars(module).items()
           if getattr(v, "__module__", None) == module.__name__ and not n.startswith("_")}
    classes = {n: v for n, v in own.items() if isinstance(v, type)}
    assert list(classes) == [f"{CAMEL[name]}Client"], f"클래스는 {CAMEL[name]}Client 하나만: {list(classes)}"
    cls = classes[f"{CAMEL[name]}Client"]
    assert issubclass(cls, LLMClient) and cls.name == module.ENGINE_NAME
    assert issubclass(cls, CategorizatorClient) == (name == "router")  # router만 분류 전용 클라이언트
    # 함수는 빌드 함수 하나, 클래스 메서드는 손질 메서드 이름만 (오타가 조용히 무시되지 않게. 헬퍼는 _로 시작)
    functions = {n for n, v in own.items() if callable(v) and not isinstance(v, type)}
    assert functions == {f"build_{name}_client"}, f"모르는 공개 함수: {functions - {f'build_{name}_client'}}"
    methods = {n for n, v in vars(cls).items() if callable(v) and not n.startswith("_")}
    assert methods <= set(HOOK_NAMES), f"모르는 메서드 이름(손질 메서드는 {HOOK_NAMES}): {methods - set(HOOK_NAMES)}"


def test_default_module_has_only_the_default_spec() -> None:
    module = engine_module("default")
    own = {n for n, v in vars(module).items()
           if getattr(v, "__module__", None) == module.__name__ and not n.startswith("_") and callable(v)}
    assert own == set() and isinstance(module.SPEC, EngineSpec) and module.OWNER == OWNERS["default"]
    with pytest.raises(ValueError, match="모르는 시나리오 엔진"):
        build_scenario_engine("default", settings=_settings())


def test_categorizator_lives_with_the_engines() -> None:
    assert CategorizatorClient.__module__ == "app.llm.engines.llm_categorizator"
    assert not importlib.util.find_spec("app.llm.llm_categorizator")  # 옛 위치는 없앴다


def test_only_router_is_a_categorizator_client(fake_tf: _FakeTransformers) -> None:
    for name in ENGINE_KEYS:
        engine = build_scenario_engine(name, settings=_settings())
        assert isinstance(engine, LLMClient)
        assert isinstance(engine, CategorizatorClient) == (name == "router"), name
        assert type(engine).__name__ == f"{CAMEL[name]}Client" and engine.name == name.upper()


def test_router_goes_through_the_same_generation(fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_hooks(monkeypatch, "router", finish_text=lambda self, text: "[라우터]" + text)
    assert _say(build_scenario_engine("router", settings=_settings())).text == "[라우터]모델 응답"


def test_llm_client_chat_needs_a_config() -> None:
    class Bare(LLMClient):
        name = "bare"

    with pytest.raises(LLMConfigError, match="config") as info:
        Bare().chat([Message(role="user", content="x")])
    assert info.value.engine == "bare"


def test_subclass_can_still_override_chat() -> None:
    class Own(LLMClient):
        name = "own"

        def chat(self, messages, *, system="", tools=None, max_tokens=None):  # noqa: ANN001
            from app.llm.llm_types import LLMResponse
            return LLMResponse(text="직접 구현")

    assert Own().chat([Message(role="user", content="x")]).text == "직접 구현"


def test_default_has_a_model() -> None:
    assert engine_module("default").SPEC.model_id  # 다른 시나리오가 None이면 이 모델을 쓴다


def test_default_spec_keeps_main_behaviour() -> None:
    # main은 모든 키가 CATEGORIZATOR + Qwen3-0.6B(cpu, 추론 끔)였다. 일부러 바꿀 때만 이 값을 같이 고친다.
    spec = engine_module("default").SPEC
    assert (spec.model_id, spec.device, spec.enable_thinking) == ("Qwen/Qwen3-0.6B", "cpu", False)
    assert not hasattr(spec, "max_new_tokens")  # 최대 생성 길이는 엔진마다 정하지 않는다 (ai_max_output_tokens)
    assert set(HOOK_NAMES) == {"prepare_messages", "generate_options", "finish_text"}


def test_unknown_engine() -> None:
    with pytest.raises(ValueError, match="모르는 시나리오 엔진"):
        engine_module("record_query")


@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_every_engine_builds_without_loading(name: str, fake_tf: _FakeTransformers) -> None:
    engine = build_scenario_engine(name, settings=_settings())
    assert isinstance(engine, _client_class(name)) and engine.name == name.upper() and engine.config.model_id
    assert fake_tf.loads == []  # 만들 때는 내려받지 않는다 (첫 chat 때 읽음)


# ── 설정 순서 (_base.resolve_config) ──

DEFAULT = EngineSpec(model_id="team/default", device="cpu", enable_thinking=False)


def test_none_fields_follow_default() -> None:
    cfg = resolve_config(EngineSpec(), DEFAULT, settings=_settings(), engine="audit_query")
    assert (cfg.model_id, cfg.device, cfg.enable_thinking) == ("team/default", "cpu", False)


def test_spec_beats_default_and_argument_beats_spec() -> None:
    spec = EngineSpec(model_id="d/own", device="cuda", enable_thinking=True, revision="abc")
    cfg = resolve_config(spec, DEFAULT, settings=_settings(), engine="audit_query")
    assert (cfg.model_id, cfg.device, cfg.enable_thinking, cfg.revision) == ("d/own", "cuda", True, "abc")
    assert resolve_config(spec, DEFAULT, settings=_settings(), model_id=" d/try ", engine="audit_query").model_id == "d/try"


def test_code_defaults_when_nothing_set() -> None:
    cfg = resolve_config(EngineSpec(), EngineSpec(model_id="m"), settings=_settings(), engine="router")
    assert (cfg.device, cfg.enable_thinking, cfg.local_files_only) == ("cpu", False, False)


def test_closed_network_forces_local_files() -> None:
    spec = EngineSpec(local_files_only=False)
    assert resolve_config(spec, DEFAULT, settings=_settings(network_env="CLOSED"), engine="router").local_files_only is True
    assert resolve_config(spec, DEFAULT, settings=_settings(), engine="router").local_files_only is False
    assert resolve_config(EngineSpec(local_files_only=True), DEFAULT, settings=_settings(), engine="router").local_files_only


def test_max_tokens_comes_from_settings_for_every_engine() -> None:
    # 엔진 설정(SPEC)에는 최대 생성 길이가 없고, 모든 엔진이 Settings.ai_max_output_tokens를 쓴다
    assert "max_new_tokens" not in EngineSpec.__dataclass_fields__
    for tokens in (1, 321, 1000):
        cfg = resolve_config(EngineSpec(), DEFAULT, settings=_settings(ai_max_output_tokens=tokens), engine="AUDIT_QUERY")
        assert cfg.max_new_tokens == tokens


@pytest.mark.parametrize("bad", [0, -5, True, "1000", None])
def test_bad_ai_max_output_tokens(bad: Any) -> None:
    class Fake:
        network_env = "OPEN"
        ai_max_output_tokens = bad

    with pytest.raises(LLMConfigError, match="AI_MAX_OUTPUT_TOKENS") as info:
        resolve_config(EngineSpec(), DEFAULT, settings=Fake(), engine="AUDIT_QUERY")
    assert info.value.engine == "AUDIT_QUERY"


def test_missing_model_is_config_error() -> None:
    with pytest.raises(LLMConfigError, match="llm_audit_query.py") as info:
        resolve_config(EngineSpec(model_id="  "), EngineSpec(), settings=_settings(), engine="AUDIT_QUERY")
    assert info.value.engine == "AUDIT_QUERY"


def test_build_error_names_the_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module("default"), "SPEC", EngineSpec())  # default에도 모델이 없으면
    with pytest.raises(LLMConfigError, match="llm_audit_query.py") as info:
        build_scenario_engine("audit_query", settings=_settings())
    assert info.value.engine == "AUDIT_QUERY"


# ── chat 흐름 (가짜 transformers) ──

def test_chat_goes_through_categorizator(fake_tf: _FakeTransformers) -> None:
    resp = _say(build_scenario_engine("audit_query", settings=_settings()))
    assert resp.text == "모델 응답" and resp.stop_reason == "end"
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (3, 2)
    messages, kw = fake_tf.template_calls[0]
    assert messages == [{"role": "system", "content": "JSON만"}, {"role": "user", "content": "분류해줘"}]
    assert kw["enable_thinking"] is False
    assert fake_tf.generate_calls[0]["max_new_tokens"] == 64 and fake_tf.generate_calls[0]["do_sample"] is False


def test_same_model_is_loaded_once_across_engines(fake_tf: _FakeTransformers) -> None:
    _say(build_scenario_engine("audit_query", settings=_settings(), model_id="team/shared"))
    _say(build_scenario_engine("daily_safety_log", settings=_settings(), model_id="team/shared"))
    _say(build_scenario_engine("router", settings=_settings(), model_id="team/other"))
    assert [m for m, _ in fake_tf.loads] == ["team/shared", "team/other"]


def test_closed_network_loads_local_only(fake_tf: _FakeTransformers) -> None:
    _say(build_scenario_engine("router", settings=_settings(network_env="CLOSED"), model_id="D:/models/q"))
    assert fake_tf.loads[0] == ("D:/models/q", {"local_files_only": True, "revision": None, "torch_dtype": "auto"})


def _set_hooks(monkeypatch: pytest.MonkeyPatch, name: str, **methods: Any) -> None:
    """그 시나리오 클라이언트 클래스의 손질 메서드를 바꿔 끼운다 (self를 받는 함수)."""
    for hook, fn in methods.items():
        assert hook in HOOK_NAMES  # 등록된 손질 메서드 이름만 쓴다
        monkeypatch.setattr(_client_class(name), hook, fn)


def test_hooks_are_used(fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_hooks(
        monkeypatch, "near_miss_warning",
        prepare_messages=lambda self, msgs: [*msgs[:-1], {"role": "user", "content": msgs[-1]["content"] + " (JSON 한 개)"}],
        generate_options=lambda self: {"repetition_penalty": 1.1},
        finish_text=lambda self, text: text.replace("모델", "[손질]"),
    )
    engine = build_scenario_engine("near_miss_warning", settings=_settings())
    assert _say(engine, "보고").text == "[손질] 응답"
    assert fake_tf.template_calls[0][0][-1]["content"] == "보고 (JSON 한 개)"
    assert fake_tf.generate_calls[0]["repetition_penalty"] == 1.1


def test_hooks_only_affect_their_own_scenario(fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_hooks(monkeypatch, "near_miss_warning", finish_text=lambda self, text: "손질됨")
    assert _say(build_scenario_engine("near_miss_warning", settings=_settings())).text == "손질됨"
    assert _say(build_scenario_engine("audit_query", settings=_settings())).text == "모델 응답"


@pytest.mark.parametrize("bad", [{"max_new_tokens": 5}, {"do_sample": True}])
def test_generate_options_cannot_override_fixed_args(
    fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch, bad: dict[str, Any]
) -> None:
    _set_hooks(monkeypatch, "audit_query", generate_options=lambda self: bad)
    with pytest.raises(LLMConfigError, match="generate_options") as info:
        _say(build_scenario_engine("audit_query", settings=_settings()))
    assert info.value.engine == "AUDIT_QUERY" and fake_tf.generate_calls == []


def test_hook_bug_is_not_hidden_as_model_failure(fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch) -> None:
    def buggy(self: Any, text: str) -> str:
        raise KeyError("팀원 코드 버그")

    _set_hooks(monkeypatch, "daily_safety_log", finish_text=buggy)
    with pytest.raises(KeyError):  # LLMError(502)로 바뀌지 않는다
        _say(build_scenario_engine("daily_safety_log", settings=_settings()))


def test_module_without_builder_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(engine_module("audit_query"), "build_audit_query_client")
    with pytest.raises(LLMConfigError, match="build_audit_query_client") as info:
        build_scenario_engine("audit_query", settings=_settings())
    assert info.value.engine == "AUDIT_QUERY"


def test_missing_model_names_the_module_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module("default"), "SPEC", EngineSpec())  # default에도 모델이 없으면
    with pytest.raises(LLMConfigError, match="llm_audit_query.py") as info:
        build_scenario_engine("audit_query", settings=_settings())
    assert info.value.engine == "AUDIT_QUERY"


def test_errors_carry_scenario_engine_name(fake_tf: _FakeTransformers, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tf.fail_load = True
    with pytest.raises(LLMError) as info:
        _say(build_scenario_engine("confined_space_gate", settings=_settings(), model_id="nope/x"))
    assert info.value.engine == "CONFINED_SPACE_GATE" and "원문" not in str(info.value)
    monkeypatch.setitem(sys.modules, "transformers", None)  # 설치 안 됨
    llm_transformers.clear_model_cache()
    with pytest.raises(LLMConfigError) as info2:
        _say(build_scenario_engine("audit_query", settings=_settings(), model_id="other/y"))
    assert info2.value.engine == "AUDIT_QUERY"


def test_plain_categorizator_still_works(fake_tf: _FakeTransformers) -> None:
    client = build_llm_client(settings=_settings(), selected_engine="CATEGORIZATOR", model_id="plain/m")
    assert client.name == "CATEGORIZATOR" and _say(client).text == "모델 응답"


# ── 팩토리·제공자 연결 ──

def test_factory_builds_scenario_engine() -> None:
    engine = build_llm_client(settings=_settings(), selected_engine="audit_query")
    assert isinstance(engine, _client_class("audit_query")) and engine.name == "AUDIT_QUERY"
    assert build_llm_client(settings=_settings(), selected_engine="audit_query", model_id="x/y").config.model_id == "x/y"


def test_factory_rejects_model_id_for_fake() -> None:
    with pytest.raises(LLMConfigError):
        build_llm_client(settings=_settings(), selected_engine="FAKE", model_id="m")


def _cat(model_id: str) -> dict[str, str]:
    return {"engine": "CATEGORIZATOR", "model_id": model_id}


def test_empty_table_means_own_engine_per_key() -> None:
    provider = LLMClientProvider(_settings(), {})
    for name in MODEL_KEYS:
        if name != "default":
            assert provider.binding_for(name) == EngineBinding(name)
    assert provider.binding_for("A") == EngineBinding("work_register")  # 디스패처는 글자로 부른다
    assert provider.get("E").name == "AUDIT_QUERY"
    assert provider.get("E") is provider.get("audit_query")


def test_config_categorizator_line_becomes_scenario_engine() -> None:
    # config.py SCENARIO_MODELS 모양 (글자 키, CATEGORIZATOR + model_id)
    table = {"default": _cat("team/default"), "router": _cat("team/router"), "A": {"engine": "FAKE"}, "B": _cat("team/b")}
    provider = LLMClientProvider.from_settings(_settings(scenario_engines=table))
    assert provider.binding_for("B") == EngineBinding("confined_space_gate", "team/b")
    assert provider.binding_for("E") == EngineBinding("audit_query", "team/default")  # 줄이 없으면 default 줄의 모델
    assert provider.binding_for("A") == EngineBinding("FAKE")
    assert provider.binding_for("router") == EngineBinding("router", "team/router")
    assert isinstance(provider.get("B"), _client_class("confined_space_gate")) and provider.get("B").config.model_id == "team/b"
    assert isinstance(provider.get("router"), _client_class("router"))
    assert provider.get("E") is not provider.get("D")  # 엔진(손질 함수)은 시나리오마다, 모델은 설정이 같으면 공유


def test_module_spec_model_beats_default_line_but_not_own_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module("audit_query"), "SPEC", EngineSpec(model_id="own/e"))
    table = {"default": _cat("team/default")}
    assert LLMClientProvider(_settings(), parse_bindings(table)).binding_for("E") == EngineBinding("audit_query", "own/e")
    table["E"] = _cat("line/e")
    assert LLMClientProvider(_settings(), parse_bindings(table)).binding_for("E") == EngineBinding("audit_query", "line/e")


def test_default_line_still_turns_everything_off() -> None:
    provider = LLMClientProvider(_settings(), {"default": EngineBinding("FAKE")})
    assert all(provider.binding_for(n) == EngineBinding("FAKE") for n in MODEL_KEYS if n != "default")


def test_line_without_engine_keeps_own_engine() -> None:
    # 줄에서 engine을 빼면 그 키의 자기 엔진 (parse_bindings의 줄임 표기)
    bindings = parse_bindings({"default": {"engine": "FAKE"}, "router": "FAKE",
                               "audit_query": {"model_id": "Qwen/Qwen3-1.7B"}})
    provider = LLMClientProvider(_settings(), bindings)
    assert provider.binding_for("audit_query") == EngineBinding("audit_query", "Qwen/Qwen3-1.7B")
    assert provider.binding_for("router") == EngineBinding("FAKE")
    assert provider.binding_for("work_register") == EngineBinding("FAKE")  # 줄이 없는 키는 default 줄


def test_key_can_borrow_another_engine() -> None:
    bindings = parse_bindings({"default": _cat("team/default"), "small_site_guide": {"engine": "work_register"}})
    provider = LLMClientProvider(_settings(), bindings)
    assert provider.get("F").name == "WORK_REGISTER"  # F가 work_register 엔진(과 그 파일의 설정)을 빌려 쓴다


def test_modules_are_imported_lazily() -> None:
    for name in MODEL_KEYS:
        sys.modules.pop(f"app.llm.engines.llm_{name}", None)
    importlib.reload(importlib.import_module("app.llm.engines"))
    LLMClientProvider.from_settings(_settings(scenario_engines={"default": {"engine": "FAKE"}}))
    assert not any(f"app.llm.engines.llm_{n}" in sys.modules for n in MODEL_KEYS if n != "default")


def test_show_scenario_models_script(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                     fake_tf: _FakeTransformers) -> None:
    import runpy
    from pathlib import Path

    import app.core as core_module

    table = {"default": _cat("team/default"), "router": {"engine": "FAKE"}, "E": _cat("D:/models/q")}
    settings = _settings(scenario_engines=table)
    monkeypatch.setattr(core_module, "get_settings", lambda: settings)
    script = Path(__file__).resolve().parents[1] / "scripts" / "show_scenario_models.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(script), run_name="__main__")
    assert info.value.code == 0
    rows = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines()[2:]}
    assert rows["router"][2] == "FAKE" and rows["audit_query"][3] == "D:/models/q"
    assert rows["daily_safety_log"][2:4] == ["daily_safety_log", "team/default"]
    assert rows["audit_query"][-1] == "AuditQueryClient" and rows["router"][-1] != "RouterClient"  # FAKE 줄에는 클래스 없음
    assert set(rows) == set(MODEL_KEYS) - {"default"}
    assert fake_tf.loads == []  # 모델을 올리지 않는다


def test_show_scenario_models_script_keeps_going_after_a_bad_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_tf: _FakeTransformers
) -> None:
    import runpy
    from pathlib import Path

    import app.core as core_module

    def broken(**_: Any) -> Any:
        raise LLMConfigError("일부러 깨뜨린 설정", engine="AUDIT_QUERY")

    monkeypatch.setattr(engine_module("audit_query"), "build_audit_query_client", broken)
    settings = _settings(scenario_engines={"default": _cat("team/default")})
    monkeypatch.setattr(core_module, "get_settings", lambda: settings)
    script = Path(__file__).resolve().parents[1] / "scripts" / "show_scenario_models.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(script), run_name="__main__")
    assert info.value.code == 0
    lines = capsys.readouterr().out.splitlines()[2:]
    rows = {line.split()[0]: line for line in lines}
    assert "오류: 일부러 깨뜨린 설정" in rows["audit_query"]
    assert rows["small_site_guide"].endswith("SmallSiteGuideClient")  # 뒤의 키도 계속 보여 준다


# ── 생성 공통 부분 (llm_transformers.generate_chat): 도구 거부, LLMError 통과 ──

def test_tools_are_rejected(fake_tf: _FakeTransformers) -> None:
    from app.llm.llm_types import ToolSpec

    engine = build_scenario_engine("audit_query", settings=_settings())
    with pytest.raises(LLMConfigError, match="도구 호출") as info:
        engine.chat([Message(role="user", content="x")], tools=[ToolSpec(name="t", description="d")])
    assert info.value.engine == "AUDIT_QUERY" and fake_tf.generate_calls == []


@pytest.mark.parametrize("message", [
    Message(role="tool", content="결과", tool_call_id="1"),
    Message(role="assistant", content="", tool_calls=[{"id": "1", "name": "t", "arguments": {}}]),
])
def test_tool_messages_are_rejected(fake_tf: _FakeTransformers, message: Message) -> None:
    engine = build_scenario_engine("audit_query", settings=_settings())
    with pytest.raises(LLMConfigError, match="도구 메시지") as info:
        engine.chat([message])
    assert info.value.engine == "AUDIT_QUERY" and fake_tf.generate_calls == []


def test_llm_error_from_inside_generation_is_not_rewrapped(fake_tf: _FakeTransformers,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kw: Any) -> None:
        raise LLMError("원래 메시지", engine="audit_query")

    engine = build_scenario_engine("audit_query", settings=_settings())
    _say(engine)  # 모델을 한 번 읽어 둔다
    loaded = next(iter(llm_transformers._cache.values()))
    monkeypatch.setattr(loaded.model, "generate", boom)
    with pytest.raises(LLMError) as info:
        _say(engine)
    assert str(info.value) == "원래 메시지" and info.value.__cause__ is None


@pytest.mark.parametrize("bad", [{"model_id": "", "max_new_tokens": 5}, {"model_id": "m", "max_new_tokens": 0}])
def test_categorizator_client_validates_its_config(bad: dict[str, Any]) -> None:
    from app.llm.llm_transformers import TransformersConfig

    with pytest.raises(ValueError):
        CategorizatorClient(TransformersConfig(**bad))


# ── config.py Engine의 대문자 시나리오 엔진 이름 ("CONFINED_SPACE_GATE" 등) ──

def test_uppercase_scenario_engine_names_in_table() -> None:
    from app.llm.engine_names import ENGINE_ALIASES, canonical_engine

    assert ENGINE_ALIASES["CONFINED_SPACE_GATE"] == "confined_space_gate" and canonical_engine("FAKE") == "FAKE"
    # config.py의 표는 FAKE가 아닌 엔진에 model_id를 요구한다 (대문자 시나리오 엔진 이름에도 model_id를 쓴다)
    table = {"default": {"engine": "FAKE"},
             "B": {"engine": "CONFINED_SPACE_GATE", "model_id": "team/b"},
             "E": {"engine": "AUDIT_QUERY", "model_id": "team/e"}}
    provider = LLMClientProvider.from_settings(_settings(scenario_engines=table))
    assert provider.binding_for("B") == EngineBinding("confined_space_gate", "team/b")
    assert provider.get("B").name == "CONFINED_SPACE_GATE" and provider.get("B").config.model_id == "team/b"
    assert provider.get("E").name == "AUDIT_QUERY" and isinstance(provider.get("E"), _client_class("audit_query"))
    assert provider.get("E").config.model_id == "team/e"
    assert provider.binding_for("A") == EngineBinding("FAKE")


def test_uppercase_names_in_table_and_factory() -> None:
    provider = LLMClientProvider(_settings(), parse_bindings({"default": {"engine": "FAKE"}, "F": "WORK_REGISTER"}))
    assert provider.binding_for("F") == EngineBinding("work_register")
    assert build_llm_client(settings=_settings(), selected_engine="DAILY_SAFETY_LOG").name == "DAILY_SAFETY_LOG"
    with pytest.raises(ValueError, match="CONFINED_SPACE_GATE"):  # 틀리면 대문자 이름도 보기에 나온다
        parse_bindings({"B": {"engine": "CONFINED"}})


# ── 모든 모듈의 빌드 함수가 같은 규칙을 지키는지 ──

def _builder(name: str) -> Any:
    return getattr(engine_module(name), f"build_{name}_client")


@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_every_builder_uses_the_given_model_id(name: str) -> None:
    client = _builder(name)(settings=_settings(), model_id=" team/x ")
    assert client.config.model_id == "team/x" and client.name == name.upper()


@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_every_builder_reports_its_engine_name_and_file(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module("default"), "SPEC", EngineSpec())
    monkeypatch.setattr(engine_module(name), "SPEC", EngineSpec())
    with pytest.raises(LLMConfigError, match=f"llm_{name}.py") as info:
        _builder(name)(settings=_settings())
    assert info.value.engine == name.upper()


@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_every_builder_uses_its_own_spec(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module(name), "SPEC", EngineSpec(model_id="own/m", device="cuda"))
    cfg = _builder(name)(settings=_settings(ai_max_output_tokens=77)).config
    assert (cfg.model_id, cfg.device, cfg.max_new_tokens) == ("own/m", "cuda", 77)  # 길이는 SPEC이 아니라 설정값


def test_default_is_not_an_engine_name() -> None:
    from app.llm.engine_names import ENGINE_ALIASES, ENGINE_NAMES, MODEL_ID_ENGINES, canonical_engine

    assert "default" not in ENGINE_NAMES and "default" not in MODEL_ID_ENGINES and "DEFAULT" not in ENGINE_ALIASES
    assert canonical_engine("DEFAULT") == "DEFAULT"
    with pytest.raises(ValueError):
        parse_bindings({"A": {"engine": "default"}})
    with pytest.raises(ValueError):
        parse_bindings({"default": {"model_id": "m"}})  # engine을 빼면 자기 엔진인데 default는 엔진이 아니다


def test_subclass_with_own_init_still_fails_cleanly_without_config() -> None:
    class Own(LLMClient):
        name = "own"

        def __init__(self) -> None:  # super().__init__()을 부르지 않는 옛 방식
            self.x = 1

    with pytest.raises(LLMConfigError, match="config"):
        Own().chat([Message(role="user", content="x")])


# ── 최대 생성 길이: ai_max_output_tokens가 모든 엔진의 상한 ──

@pytest.mark.parametrize("name", ENGINE_KEYS)
def test_every_engine_uses_ai_max_output_tokens_as_the_limit(name: str, fake_tf: _FakeTransformers) -> None:
    engine = build_scenario_engine(name, settings=_settings(ai_max_output_tokens=300))
    engine.chat([Message(role="user", content="x")])  # 호출자가 길이를 안 주면 설정값
    engine.chat([Message(role="user", content="x")], max_tokens=5000)  # 더 크게 요청해도 설정값이 상한
    engine.chat([Message(role="user", content="x")], max_tokens=200)  # 더 작게 요청하면(라우터 등) 그 값
    assert [c["max_new_tokens"] for c in fake_tf.generate_calls] == [300, 300, 200]


def test_plain_categorizator_also_uses_ai_max_output_tokens(fake_tf: _FakeTransformers) -> None:
    client = build_llm_client(settings=_settings(ai_max_output_tokens=123), selected_engine="CATEGORIZATOR", model_id="plain/m")
    assert client.config.max_new_tokens == 123
    _say(client)
    assert fake_tf.generate_calls[0]["max_new_tokens"] == 64  # 이 테스트 helper가 max_tokens=64를 줌 (123보다 작아 그대로)


def test_stop_reason_follows_the_effective_limit(fake_tf: _FakeTransformers) -> None:
    # 가짜 모델은 새 토큰 2개를 만든다. 상한이 2이면 eos가 있어도 끝난 것, 상한이 1이면...
    engine = build_scenario_engine("audit_query", settings=_settings(ai_max_output_tokens=2))
    assert engine.chat([Message(role="user", content="x")]).stop_reason == "end"
