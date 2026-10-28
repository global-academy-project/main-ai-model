"""시나리오별 LLM 엔진·모델 제공자.

`SCENARIO_ENGINES`(환경 변수, JSON)로 시나리오마다 어떤 엔진·모델을 쓸지 정한다.

    SCENARIO_ENGINES='{
      "router":  {"engine": "CATEGORIZATOR", "model_id": "작은-분류-모델"},
      "A":       {"engine": "CATEGORIZATOR", "model_id": "큰-생성-모델"},
      "F":       {"engine": "CATEGORIZATOR", "model_id": "큰-생성-모델"},
      "default": {"engine": "FAKE"}
    }'

키: `router`(시나리오 분류용), `A`~`F`(시나리오별), `default`(지정 안 한 키).
`default`도 없으면 `DEFAULT_LLM_ENGINE`을 쓴다. 문자열 하나("FAKE")도 허용한다.

규칙:
- 엔진·모델은 **서버 설정으로만** 정한다. 요청자가 고를 수 없다 (요청 본문에 엔진 이름 필드를 두지 않는다).
- 같은 (엔진, 모델)은 한 번만 만들어 시나리오끼리 공유한다 (transformers 모델이 메모리에 중복으로 올라가지 않게).
- 설정 오타·허용되지 않은 값·`NETWORK_ENV`에서 막힌 엔진은 **기동 때** `ValueError`.
  기동 뒤 `get()`에서는 `LLMError` 계열만 나온다 (`EngineBlockedError`, `LLMConfigError`).
- `model_id` 지정은 CATEGORIZATOR 엔진만 지원한다 (OPENAI 엔진은 아직 미구현).
- `llm` 계층이라 `agent`를 모른다. 키는 시나리오 코드("A"~"F") 문자열이다.
"""
from __future__ import annotations
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, get_args

from app.core import Engine

from .llm_base import LLMClient

ROUTER_KEY = "router"
DEFAULT_KEY = "default"
SCENARIO_KEYS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
VALID_KEYS: tuple[str, ...] = (ROUTER_KEY, DEFAULT_KEY, *SCENARIO_KEYS)
_ENGINES: tuple[str, ...] = get_args(Engine)
_BINDING_FIELDS = {"engine", "model_id"}

@dataclass(frozen=True)
class EngineBinding:
    engine: str
    model_id: str | None = None  # None이면 엔진의 기본 모델 설정을 쓴다

def parse_bindings(raw: str | Mapping[str, Any] | None) -> dict[str, EngineBinding]:
    """`SCENARIO_ENGINES` 값을 검증해 바인딩으로 바꾼다. 잘못되면 ValueError (기동 실패)."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("SCENARIO_ENGINES가 올바른 JSON이 아닙니다.") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("SCENARIO_ENGINES는 JSON 객체여야 합니다.")

    bindings: dict[str, EngineBinding] = {}
    for key, value in raw.items():
        if key not in VALID_KEYS:
            raise ValueError(f"SCENARIO_ENGINES에 모르는 키가 있습니다: {key!r} (허용: {', '.join(VALID_KEYS)})")
        if isinstance(value, str):
            value = {"engine": value}
        if not isinstance(value, Mapping):
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]는 객체 또는 엔진 이름이어야 합니다.")
        if unknown := set(value) - _BINDING_FIELDS:
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]에 모르는 항목이 있습니다: {sorted(unknown)}")
        engine = value.get("engine")
        if engine not in _ENGINES:
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]의 engine은 {', '.join(_ENGINES)} 중 하나여야 합니다.")
        model_id = value.get("model_id")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]의 model_id가 비어 있습니다.")
        if model_id and engine != "CATEGORIZATOR":
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]: model_id 지정은 CATEGORIZATOR 엔진만 지원합니다.")
        bindings[key] = EngineBinding(engine, model_id.strip() if model_id else None)
    return bindings


ClientFactory = Callable[[str, "str | None"], LLMClient]

class LLMClientProvider:
    def __init__(
        self,
        settings: Any,
        bindings: Mapping[str, EngineBinding] | None = None,
        *,
        factory: ClientFactory | None = None,
    ) -> None:
        self._settings = settings
        self._bindings = dict(bindings or {})
        for key, binding in self._bindings.items():
            if settings.is_engine_blocked(binding.engine):
                raise ValueError(
                    f"시나리오 '{key}'의 엔진 '{binding.engine}'은(는) 현재 NETWORK_ENV에서 사용할 수 없습니다."
                )
        self._factory = factory or self._default_factory
        self._clients: dict[tuple[str, str | None], LLMClient] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Any) -> LLMClientProvider:
        return cls(settings, parse_bindings(getattr(settings, "scenario_engines", None)))

    def binding_for(self, key: str) -> EngineBinding:
        if key not in VALID_KEYS or key == DEFAULT_KEY:
            raise ValueError(f"모르는 시나리오 키: {key!r}")
        return (
            self._bindings.get(key)
            or self._bindings.get(DEFAULT_KEY)
            or EngineBinding(self._settings.default_llm_engine)
        )

    def get(self, key: str) -> LLMClient:
        """시나리오(또는 router)에 맞는 엔진. 같은 (엔진, 모델)이면 같은 객체를 돌려준다.

        `EngineBlockedError`·`LLMConfigError`는 여기서 난다 → 호출하는 쪽은 `try/except LLMError` 안에서 부른다.
        """
        binding = self.binding_for(key)
        cache_key = (binding.engine, binding.model_id)
        with self._lock:
            client = self._clients.get(cache_key)
            if client is None:
                client = self._factory(binding.engine, binding.model_id)
                self._clients[cache_key] = client
            return client

    def describe(self) -> dict[str, dict[str, str | None]]:
        """키별로 실제 쓰이는 엔진·모델 (진단용. 키·비밀값은 들어 있지 않다)."""
        return {
            key: {"engine": b.engine, "model_id": b.model_id}
            for key in (ROUTER_KEY, *SCENARIO_KEYS)
            for b in (self.binding_for(key),)
        }

    def _default_factory(self, engine: str, model_id: str | None) -> LLMClient:
        # 엔진을 고르는 유일한 지점은 llm_factory다 (폐쇄망 차단·설정 검사도 거기서 한다).
        from .llm_factory import build_llm_client

        return build_llm_client(settings=self._settings, selected_engine=engine, model_id=model_id)  # type: ignore[arg-type]
