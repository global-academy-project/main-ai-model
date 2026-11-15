"""모델별 생성 잠금 + 기동 시 모델 미리 올리기 (11/15). 실제 모델·torch 없이 가짜 토크나이저·모델로 확인한다."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.exceptions import LLMConfigError, LLMError
from app.llm import llm_transformers as lt
from app.llm.llm_base import LLMClient
from app.llm.llm_preload import PRELOAD_KEYS, preload_models
from app.llm.llm_transformers import TransformersConfig
from app.llm.llm_types import Message

EOS = 99


class _Ids(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, index):
        item = list.__getitem__(self, index)
        return _Ids(item) if isinstance(index, slice) else item


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = EOS

    def apply_chat_template(self, messages, **kwargs):
        return _Inputs(input_ids=_Ids([1, 2, 3]))

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class _Model:
    """generate 안에 동시에 몇 개가 들어와 있는지 센다 (전 모델 공통 카운터)."""

    device = "cpu"

    def __init__(self, meter, barrier=None):
        self.meter = meter
        self.barrier = barrier

    def generate(self, input_ids, max_new_tokens, do_sample, **kwargs):
        self.meter.enter()
        try:
            if self.barrier is not None:
                self.barrier.wait()  # 두 생성이 동시에 들어와야 통과 (아니면 BrokenBarrierError)
            else:
                time.sleep(0.05)
        finally:
            self.meter.leave()
        return [_Ids([*input_ids, 7, EOS])]


class _Meter:
    def __init__(self):
        self._lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def enter(self):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def leave(self):
        with self._lock:
            self.now -= 1


def _config(model_id):
    return TransformersConfig(model_id=model_id, max_new_tokens=8)


@pytest.fixture
def fake_models(monkeypatch):
    """캐시를 비우고, 잠금 범위 설정을 고를 수 있게 get_settings 를 바꾼다."""
    monkeypatch.delenv("LLM_GENERATE_LOCK", raising=False)
    monkeypatch.delenv("LLM_PRELOAD", raising=False)
    lt.clear_model_cache()
    scope = {"value": "MODEL"}
    monkeypatch.setattr("app.core.get_settings", lambda: SimpleNamespace(llm_generate_lock=scope["value"], llm_adapters={}))
    yield scope
    lt.clear_model_cache()


def _put(config, model):
    lt._cache[config] = lt._LoadedModel(_Tokenizer(), model)


def _run_two(cfg_a, cfg_b):
    errors = []

    def call(cfg):
        try:
            lt.generate_chat(LLMClient(cfg), cfg, [Message(role="user", content="hi")])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(c,)) for c in (cfg_a, cfg_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    return errors


def test_different_models_generate_at_the_same_time(fake_models):
    meter, barrier = _Meter(), threading.Barrier(2, timeout=3)
    small, large = _config("Qwen/Qwen3-0.6B"), _config("Qwen/Qwen3-1.7B")
    _put(small, _Model(meter, barrier))
    _put(large, _Model(meter, barrier))
    assert _run_two(small, large) == []
    assert meter.peak == 2


def test_same_model_still_one_at_a_time(fake_models):
    meter = _Meter()
    small = _config("Qwen/Qwen3-0.6B")
    _put(small, _Model(meter))
    assert _run_two(small, small) == []
    assert meter.peak == 1


def test_global_switch_restores_old_behavior(fake_models):
    fake_models["value"] = "GLOBAL"
    meter = _Meter()
    small, large = _config("Qwen/Qwen3-0.6B"), _config("Qwen/Qwen3-1.7B")
    _put(small, _Model(meter))
    _put(large, _Model(meter))
    assert _run_two(small, large) == []
    assert meter.peak == 1


def test_lock_scope_defaults_to_model_when_settings_unreadable(monkeypatch):
    def broken():
        raise RuntimeError("no env")

    monkeypatch.setattr("app.core.get_settings", broken)
    loaded = lt._LoadedModel(_Tokenizer(), _Model(_Meter()))
    assert lt._generate_lock_for(loaded) is loaded.lock


def test_settings_defaults(monkeypatch):
    from app.core.config import Settings

    for name in ("LLM_PRELOAD", "LLM_GENERATE_LOCK"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK")
    assert settings.llm_preload is True
    assert settings.llm_generate_lock == "MODEL"
    monkeypatch.setenv("LLM_PRELOAD", "false")
    monkeypatch.setenv("LLM_GENERATE_LOCK", "GLOBAL")
    settings = Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK")
    assert settings.llm_preload is False
    assert settings.llm_generate_lock == "GLOBAL"


# ── 기동 시 미리 올리기 ──

class _Provider:
    def __init__(self, clients):
        self.clients = clients
        self.asked = []

    def get(self, key):
        self.asked.append(key)
        value = self.clients.get(key)
        if isinstance(value, Exception):
            raise value
        return value


def _client(name, config):
    client = LLMClient(config)
    client.name = name
    return client


def test_preload_reads_each_model_once_and_skips_fake():
    small, large = _config("Qwen/Qwen3-0.6B"), _config("Qwen/Qwen3-1.7B")
    clients = {key: _client(key.upper(), small) for key in PRELOAD_KEYS}
    clients["daily_safety_log"] = _client("DAILY_SAFETY_LOG", large)
    clients["small_site_guide"] = LLMClient()  # config 없음 (FAKE 와 같음)
    loaded = []
    results = preload_models(_Provider(clients), load=lambda cfg, engine: loaded.append(cfg.model_id))
    assert sorted(loaded) == ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    by_model = {r.model_id: r for r in results}
    assert by_model["Qwen/Qwen3-1.7B"].keys == ("daily_safety_log",)
    assert "router" in by_model["Qwen/Qwen3-0.6B"].keys
    assert "small_site_guide" not in by_model["Qwen/Qwen3-0.6B"].keys
    assert all(r.ok for r in results)


def test_preload_failure_does_not_stop_and_others_still_load():
    small, large = _config("Qwen/Qwen3-0.6B"), _config("broken/model")
    clients = {key: _client(key.upper(), small) for key in PRELOAD_KEYS}
    clients["daily_safety_log"] = _client("DAILY_SAFETY_LOG", large)
    clients["audit_query"] = LLMConfigError("막힘", engine="AUDIT_QUERY")  # provider.get 이 오류

    def load(cfg, engine):
        if cfg.model_id == "broken/model":
            raise LLMError("모델을 불러오지 못했습니다.", engine=engine)

    results = {r.model_id: r for r in preload_models(_Provider(clients), load=load)}
    assert results["broken/model"].ok is False
    assert results["Qwen/Qwen3-0.6B"].ok is True
    assert "audit_query" not in results["Qwen/Qwen3-0.6B"].keys


def test_preload_with_real_cache_then_requests_reuse_it(fake_models, monkeypatch):
    """ensure_loaded 가 _load 를 거쳐 캐시에 넣고, 요청 때 다시 읽지 않는다."""
    small = _config("Qwen/Qwen3-0.6B")
    calls = []

    def fake_load(config, engine=lt.DEFAULT_ENGINE_NAME):
        calls.append(config)
        return lt._cache.setdefault(config, lt._LoadedModel(_Tokenizer(), _Model(_Meter())))

    monkeypatch.setattr(lt, "_load", fake_load)
    preload_models(_Provider({key: _client(key.upper(), small) for key in PRELOAD_KEYS}))
    assert calls == [small]
    lt.generate_chat(LLMClient(small), small, [Message(role="user", content="hi")])
    assert lt._cache[small].model is not None and len(calls) == 2  # 요청도 _load 를 부르지만 캐시에서 돌려준다


# ── create_app 기동 절차 ──

@pytest.mark.parametrize("flag, expected", [("true", 1), ("false", 0)])
def test_lifespan_calls_preload_only_when_enabled(monkeypatch, flag, expected):
    from fastapi.testclient import TestClient

    from app.core import get_settings

    monkeypatch.setenv("NETWORK_ENV", "OPEN")
    monkeypatch.setenv("CHECK_RULES_TOOL", "MOCK")
    monkeypatch.setenv("LLM_PRELOAD", flag)
    get_settings.cache_clear()
    from app import main  # 처음 임포트할 때 모듈 끝의 create_app()이 설정을 읽으므로 환경변수 뒤에 둔다
    seen = []
    monkeypatch.setattr(main, "preload_models", lambda provider: seen.append(provider))
    try:
        app = main.create_app()
        with TestClient(app):
            pass
        assert len(seen) == expected
        if expected:
            assert seen[0] is app.state.llm_provider
    finally:
        get_settings.cache_clear()
