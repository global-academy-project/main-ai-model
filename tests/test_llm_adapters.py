"""LoRA 어댑터 분기 테스트 (llm_transformers._use_adapter). torch·transformers·peft 없이 돈다.

같은 기본 모델 하나에 어댑터를 이름별로 붙이고, 요청한 엔진의 어댑터로 바꿔 생성하는지 본다.
어댑터를 쓰지 않는 엔진·모델은 지금까지와 똑같이 동작해야 한다.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any

import pytest

from app.llm import llm_transformers
from app.llm.llm_base import LLMClient
from app.llm.llm_transformers import TransformersConfig
from app.llm.llm_types import Message
from tests.test_scenario_engines import _FakeTransformers


class _FakePeft:
    """peft.PeftModel 흉내: 기본 모델을 감싸고, 이름별 어댑터 중 지금 켜진 것을 기록한다."""

    calls: list[tuple[str, str, str]] = []

    def __init__(self, base: Any, path: str, name: str) -> None:
        self.base = base
        self.device = "cpu"
        self.paths = {name: path}
        self.active: str | None = name
        self.used: list[str | None] = []

    @classmethod
    def from_pretrained(cls, model: Any, path: str, adapter_name: str) -> "_FakePeft":
        cls.calls.append(("from_pretrained", path, adapter_name))
        if "broken" in path:
            raise ValueError("어댑터 파일이 깨짐")
        return cls(model, path, adapter_name)

    def load_adapter(self, path: str, adapter_name: str) -> None:
        type(self).calls.append(("load_adapter", path, adapter_name))
        if "broken" in path:
            raise ValueError("어댑터 파일이 깨짐")
        self.paths[adapter_name] = path

    def set_adapter(self, name: str) -> None:
        assert name in self.paths
        self.active = name

    @contextmanager
    def disable_adapter(self) -> Any:
        before, self.active = self.active, None
        try:
            yield
        finally:
            self.active = before

    def eval(self) -> None:
        pass

    def generate(self, **kw: Any) -> Any:
        self.used.append(self.active)
        return self.base.generate(**kw)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> dict[str, Any]:
    tf = _FakeTransformers()
    monkeypatch.setitem(sys.modules, "transformers", tf.module)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=_FakePeft))
    _FakePeft.calls = []
    folders = {}
    for name in ("extract", "record_query"):
        (tmp_path / name).mkdir()
        folders[name] = str(tmp_path / name)
    folders["broken"] = str(tmp_path / "broken")
    (tmp_path / "broken").mkdir()
    folders["missing"] = str(tmp_path / "no-such-folder")
    monkeypatch.setattr(llm_transformers, "_adapter_paths", lambda: dict(folders))
    llm_transformers.clear_model_cache()
    yield {"tf": tf, "folders": folders}
    llm_transformers.clear_model_cache()


def _client(adapter: str | None, model_id: str = "Qwen/Qwen3-1.7B") -> LLMClient:
    client = LLMClient(TransformersConfig(model_id=model_id, max_new_tokens=32))
    client.adapter = adapter
    return client


def _say(client: LLMClient) -> str:
    return client.chat([Message(role="user", content="산소 20.9")], system="JSON만").text


def _model(model_id: str = "Qwen/Qwen3-1.7B") -> Any:
    return llm_transformers._cache[TransformersConfig(model_id=model_id, max_new_tokens=32)].model


def test_no_adapter_is_unchanged(fake: dict[str, Any]) -> None:
    assert _say(_client(None)) == "모델 응답"
    assert _FakePeft.calls == [] and not isinstance(_model(), _FakePeft)   # peft 를 거치지 않는다


def test_adapters_share_one_base_and_switch_per_request(fake: dict[str, Any]) -> None:
    extract, record, plain = _client("extract"), _client("record_query"), _client(None)
    _say(extract)
    _say(record)
    _say(plain)      # 같은 모델에서 어댑터가 없는 엔진은 어댑터를 끈다
    _say(extract)
    assert [m for m, _ in fake["tf"].loads] == ["Qwen/Qwen3-1.7B"]            # 기본 모델은 한 번만
    assert [c[0] for c in _FakePeft.calls] == ["from_pretrained", "load_adapter"]   # 어댑터도 한 번씩만
    assert _model().used == ["extract", "record_query", None, "extract"]
    assert _model().active == "extract"   # disable_adapter 뒤에는 원래대로 돌아온다


def test_missing_or_broken_adapter_falls_back_to_base(fake: dict[str, Any]) -> None:
    assert _say(_client("missing")) == "모델 응답"          # 폴더 없음 → 기본 모델
    assert _say(_client("not-configured")) == "모델 응답"   # LLM_ADAPTERS 에 없음 → 기본 모델
    assert _FakePeft.calls == [] and not isinstance(_model(), _FakePeft)
    assert _say(_client("broken")) == "모델 응답"           # 붙이다 실패 → 기본 모델, 모델은 그대로
    assert not isinstance(_model(), _FakePeft)
    _say(_client("broken"))
    assert [c[0] for c in _FakePeft.calls] == ["from_pretrained"]   # 실패한 어댑터는 다시 시도하지 않는다


def test_failed_adapter_after_another_uses_base_not_the_other(fake: dict[str, Any]) -> None:
    _say(_client("extract"))
    _say(_client("broken"))
    assert _model().used == ["extract", None]   # 실패한 어댑터 요청에 다른 엔진 어댑터를 쓰지 않는다


def test_other_model_is_not_touched(fake: dict[str, Any]) -> None:
    _say(_client("extract"))
    _say(_client(None, model_id="Qwen/Qwen3-0.6B"))
    assert not isinstance(_model("Qwen/Qwen3-0.6B"), _FakePeft)   # 어댑터 엔진이 없는 다른 기본 모델은 peft 를 거치지 않는다


def test_peft_not_installed_falls_back(fake: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "peft", None)   # import peft → ImportError
    assert _say(_client("extract")) == "모델 응답"
    assert not isinstance(_model(), _FakePeft)
