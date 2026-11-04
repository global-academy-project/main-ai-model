"""시나리오별 LLM 엔진·모델 제공자.

시나리오 키 → 엔진. 팀 표는 `app/core/config.py`의 `SCENARIO_MODELS`(= Settings.scenario_engines)이고,
시나리오 엔진 모듈은 app/llm/engines/llm_<키>.py다 (엔진 이름 = 시나리오 키).

키 K의 엔진을 정하는 순서
  1. 표에 K 줄이 있으면 그 값
  2. 없고 `default` 줄이 있으면 그 값 (테스트에서 {"default": FAKE}로 전부 끄는 경로)
  3. 둘 다 없으면 K와 같은 이름의 자기 엔진 (예: work_register → llm_work_register.py)
표의 `CATEGORIZATOR`(transformers 로컬 모델)는 **K의 시나리오 엔진 모듈로** 만든다 (router면 분류 전용
RouterEngine, 나머지는 LLMClient 기반 ScenarioEngine). 모델 id는
  K 줄의 model_id → (default 줄을 따를 때) 모듈 SPEC의 model_id → default 줄의 model_id → llm_default.py SPEC
순이다. 장치·손질 메서드는 모듈에서 정하고, 최대 생성 길이는 모든 엔진이 ai_max_output_tokens 하나를 쓴다.

줄 값: 엔진 이름 문자열("FAKE") 또는 {"engine", "model_id"}. engine을 빼면 그 키의 자기 엔진이다.
키: `router`, 시나리오 이름(`app/core/scenario_keys.py`: work_register·confined_space_gate·near_miss_warning·
daily_safety_log·audit_query·small_site_guide), `default`. 글자 코드("A"~"F")도 별칭으로 받는다
(분류기·디스패처가 `get("A")`로 불러도 같은 엔진). 같은 시나리오를 글자와 이름으로 함께 적으면 ValueError.

규칙:
- 엔진·모델은 **서버 설정으로만** 정한다. 요청자가 고를 수 없다 (요청 본문에 엔진 이름 필드를 두지 않는다).
- 같은 (엔진, 모델)은 한 번만 만들어 시나리오끼리 공유한다 (transformers 모델이 메모리에 중복으로 올라가지 않게).
- 설정 오타·허용되지 않은 값·`NETWORK_ENV`에서 막힌 엔진은 **기동 때** `ValueError`.
  기동 뒤 `get()`에서는 `LLMError` 계열만 나온다 (`EngineBlockedError`, `LLMConfigError`).
- `model_id` 지정은 transformers 엔진(CATEGORIZATOR·시나리오 엔진)만 지원한다 (OPENAI는 미구현 예시).
- 같은 (엔진, 모델)은 한 번 만들고, 같은 모델 설정이면 엔진이 달라도 transformers 모델은 한 번만 읽는다.
- `llm` 계층이라 `agent`를 모른다. 키는 시나리오 이름(또는 글자 코드) 문자열이다.
"""
from __future__ import annotations
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app.core.scenario_keys import (
    DEFAULT_KEY,
    MODEL_KEYS,
    ROUTER_KEY,
    SCENARIO_KEY_NAMES,
    canonical_key,
    canonical_mapping,
)

from .engine_names import ENGINE_ALIASES, ENGINE_NAMES, MODEL_ID_ENGINES, TRANSFORMERS_ENGINE, canonical_engine
from .llm_base import LLMClient

# 키 이름의 기준은 app/core/scenario_keys.py 한 곳이다 (여기서는 다시 내보내기만 한다)
SCENARIO_KEYS: tuple[str, ...] = SCENARIO_KEY_NAMES
VALID_KEYS: tuple[str, ...] = MODEL_KEYS
_ENGINES: tuple[str, ...] = ENGINE_NAMES
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
    raw = canonical_mapping(raw)  # 글자 코드 → 이름, 모르는 키·중복은 ValueError

    bindings: dict[str, EngineBinding] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            value = {"engine": value}
        if not isinstance(value, Mapping):
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]는 객체 또는 엔진 이름이어야 합니다.")
        if unknown := set(value) - _BINDING_FIELDS:
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]에 모르는 항목이 있습니다: {sorted(unknown)}")
        engine = canonical_engine(value.get("engine") or key)  # engine을 빼면 그 키의 자기 엔진. 대문자 이름도 받는다
        if engine not in _ENGINES:
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]의 engine은 {', '.join((*_ENGINES, *ENGINE_ALIASES))} 중 하나여야 합니다.")
        model_id = value.get("model_id")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]의 model_id가 비어 있습니다.")
        if model_id and engine not in MODEL_ID_ENGINES:
            raise ValueError(f"SCENARIO_ENGINES[{key!r}]: model_id 지정은 transformers 엔진(CATEGORIZATOR·시나리오 엔진)만 지원합니다.")
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
        self._bindings = canonical_mapping(dict(bindings or {}), source="bindings")
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
        """표(settings.scenario_engines)로 만든다. 막힌 엔진·틀린 값은 여기서 ValueError."""
        return cls(settings, parse_bindings(getattr(settings, "scenario_engines", None)))

    def binding_for(self, key: str) -> EngineBinding:
        name = canonical_key(key)  # 글자 코드("A")로 불러도 이름으로 바꾼다. 모르는 키는 ValueError
        if name == DEFAULT_KEY:
            raise ValueError(f"모르는 시나리오 키: {key!r} (default는 직접 받지 않는다)")
        own = self._bindings.get(name)
        binding = own or self._bindings.get(DEFAULT_KEY) or EngineBinding(name)  # 줄이 없으면 자기 엔진
        if binding.engine != TRANSFORMERS_ENGINE:
            return binding
        # 표의 CATEGORIZATOR = transformers 로컬 모델 → 이 키의 시나리오 엔진 모듈로 만든다
        model_id = binding.model_id
        if own is None:  # default 줄을 따른 경우: 내 모듈 SPEC에 model_id가 있으면 그것이 먼저
            model_id = _module_model_id(name) or model_id
        return EngineBinding(name, model_id)

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


def _module_model_id(name: str) -> str | None:
    """시나리오 모듈 SPEC의 model_id (없으면 None). 모듈은 이때 처음 임포트한다."""
    from .engines import engine_module

    spec = getattr(engine_module(name), "SPEC", None)
    model_id = getattr(spec, "model_id", None)
    return model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
