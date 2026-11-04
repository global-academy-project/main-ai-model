"""엔진 이름표 (llm 계층). `app/core/config.py`를 고치지 않고 시나리오 엔진을 더하려고 여기에 둔다.

- config의 `Engine`(FAKE·OPENAI·CATEGORIZATOR)은 시나리오 모델 표(`SCENARIO_MODELS`)에 쓰는 이름이다.
- 시나리오 엔진 이름 = 시나리오 키(router·work_register·…·default, app/core/scenario_keys.py). 모듈은
  app/llm/engines/llm_<키>.py. 표에는 쓰지 않고 코드 안에서만 쓴다.
- 표의 `CATEGORIZATOR`(transformers 로컬 모델)는 그 키의 시나리오 엔진으로 만든다 (llm_provider 참고).
- 표에 대문자 시나리오 엔진 이름(`"engine": "CONFINED_SPACE_GATE"`)을 쓰면 같은 이름의 시나리오 엔진이다. 그때 모델은
  그 모듈의 SPEC(없으면 llm_default.py)에서 정한다.
"""

from __future__ import annotations

from typing import get_args

from app.core import Engine
from app.core.scenario_keys import DEFAULT_KEY, MODEL_KEYS

ENGINE_KEYS: tuple[str, ...] = tuple(k for k in MODEL_KEYS if k != DEFAULT_KEY)  # default는 엔진이 아니라 기본 설정

TRANSFORMERS_ENGINE = "CATEGORIZATOR"  # 표에서 "transformers 로컬 모델"을 뜻하는 이름
BASE_ENGINES: tuple[str, ...] = ("FAKE", "OPENAI", TRANSFORMERS_ENGINE)  # llm_factory가 아는 엔진 (OPENAI는 미구현 예시)
CONFIG_ENGINES: tuple[str, ...] = tuple(get_args(Engine))
# 표에서 시나리오 엔진을 대문자로 쓴 이름("CONFINED_SPACE_GATE") → 엔진 이름("confined_space_gate")
ENGINE_ALIASES: dict[str, str] = {name.upper(): name for name in ENGINE_KEYS}
ENGINE_NAMES: tuple[str, ...] = tuple(dict.fromkeys((*BASE_ENGINES, *ENGINE_KEYS)))


def canonical_engine(engine: str) -> str:
    """대문자 시나리오 엔진 이름은 소문자 엔진 이름으로. 나머지는 그대로."""
    return ENGINE_ALIASES.get(engine, engine)


# model_id를 받는 엔진: transformers 엔진(CATEGORIZATOR와 시나리오 엔진)만
MODEL_ID_ENGINES: frozenset[str] = frozenset({TRANSFORMERS_ENGINE, *ENGINE_KEYS})
