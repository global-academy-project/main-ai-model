"""시나리오 엔진 등록표. 엔진 이름 = 시나리오 키 (app/core/scenario_keys.py).

엔진 모듈은 모두 이 폴더에 있다. 팀원이 고치는 파일은 **자기 `llm_<키>.py` 하나**다.

| 엔진 | 모듈 | 계획서 | 주인 |
| --- | --- | --- | --- |
| router              | llm_router.py              | —  | B |
| work_register       | llm_work_register.py       | A  | B |
| confined_space_gate | llm_confined_space_gate.py | B  | C |
| near_miss_warning   | llm_near_miss_warning.py   | C  | A |
| daily_safety_log    | llm_daily_safety_log.py    | D  | D |
| audit_query         | llm_audit_query.py         | E  | D |
| small_site_guide    | llm_small_site_guide.py    | F  | B |
| default             | llm_default.py             | —  | 공용 |

나머지 파일은 B가 맡는다 (고칠 일이 있으면 B에게):
  llm_categorizator.py  분류 전용 CategorizatorClient (router와 기본 CATEGORIZATOR 엔진만 쓴다)
  _base.py              EngineSpec · 설정 순서 · build_config (각 모듈의 빌드 함수가 부름)
(transformers 모델 읽기·생성·오류 처리는 ../llm_transformers.py, LLMClient.chat()이 부른다)
  __init__.py           이 등록표

시나리오 모듈이 가지는 것: `ENGINE_NAME`(config.py의 `Engine` 이름, 대문자), `OWNER`, `SPEC`(EngineSpec), 그 이름을 딴
클라이언트 클래스(`WorkRegisterClient` 등, `LLMClient` 상속. router만 `CategorizatorClient`), 빌드 함수 `build_<키>_client`
(예: `build_work_register_client`). 손질은 클래스의 `prepare_messages`·`generate_options`·`finish_text` 재정의로 한다.
`llm_default.py`는 엔진이 아니라 기본 SPEC만 가진다.
모듈은 쓸 때 임포트한다 (기동 때 8개를 다 불러오지 않는다). 모듈 하나가 깨져도(임포트 오류) 그 엔진을 쓰는 요청만 실패한다.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from app.core.scenario_keys import DEFAULT_KEY, MODEL_KEYS
from app.exceptions import LLMConfigError

from ..llm_base import LLMClient

# 엔진 이름 → 모듈 이름. 키 목록은 scenario_keys.MODEL_KEYS와 같아야 한다 (테스트가 지킨다)
ENGINE_MODULES: dict[str, str] = {name: f"llm_{name}" for name in MODEL_KEYS}
SCENARIO_ENGINE_NAMES: tuple[str, ...] = tuple(name for name in ENGINE_MODULES if name != DEFAULT_KEY)  # default는 엔진이 아님


def engine_module(name: str) -> ModuleType:
    if name not in ENGINE_MODULES:
        raise ValueError(f"모르는 시나리오 엔진 {name!r} (엔진: {', '.join(ENGINE_MODULES)})")
    return importlib.import_module(f"{__name__}.{ENGINE_MODULES[name]}")


def builder_name(name: str) -> str:
    """엔진 이름(소문자 키) → 빌드 함수 이름. 예: work_register → build_work_register_client"""
    return f"build_{name}_client"


def build_scenario_engine(name: str, *, settings: Any, model_id: str | None = None) -> LLMClient:
    """엔진 이름으로 그 모듈의 `build_<키>_client`를 불러 클라이언트를 만든다. 설정이 틀리면 LLMConfigError (모델은 아직 읽지 않는다)."""
    if name not in SCENARIO_ENGINE_NAMES:
        raise ValueError(f"모르는 시나리오 엔진 {name!r} (엔진: {', '.join(SCENARIO_ENGINE_NAMES)})")
    module = engine_module(name)
    builder = getattr(module, builder_name(name), None)
    if not callable(builder):
        raise LLMConfigError(f"llm_{name}.py에 {builder_name(name)}() 함수가 없습니다.", engine=name.upper())
    return builder(settings=settings, model_id=model_id)
