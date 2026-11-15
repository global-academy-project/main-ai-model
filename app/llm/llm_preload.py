"""기동 때 LLM 모델을 미리 올린다 (11/15, B). `Settings.llm_preload`(LLM_PRELOAD, 기본 True)가 False면 부르지 않는다.

- 대상: router 와 모든 시나리오 키. provider.get()으로 클라이언트를 만들고 transformers 설정(config)이 있는 것만 올린다.
  같은 설정은 한 번만 읽는다 (llm_transformers 캐시: 같은 0.6B 를 쓰는 router·A·B·E 는 한 벌). FAKE 엔진은 건너뛴다.
- LoRA 어댑터는 붙이지 않는다. 지금처럼 그 엔진을 처음 쓸 때 붙인다 (11/15 결정).
- 실패해도 서버는 뜬다 (11/15 결정): 경고만 남기고, 그 모델은 요청 때 다시 읽으려 하며 실패하면 지금과 같은 오류(LLMError)가 난다.
  읽기 실패의 원문(traceback)은 llm_transformers._load 가 이미 로그에 남긴다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core.scenario_keys import ROUTER_KEY, SCENARIO_KEY_NAMES
from app.exceptions import LLMError

from .llm_transformers import TransformersConfig, ensure_loaded

logger = logging.getLogger(__name__)

PRELOAD_KEYS: tuple[str, ...] = (ROUTER_KEY, *SCENARIO_KEY_NAMES)


@dataclass(frozen=True)
class PreloadResult:
    model_id: str
    keys: tuple[str, ...]  # 이 모델을 쓰는 키 (router, work_register, ...)
    ok: bool
    seconds: float


def preload_models(
    provider: Any,
    *,
    load: Callable[[TransformersConfig, str], None] = ensure_loaded,
) -> list[PreloadResult]:
    """provider 의 모든 키 모델을 한 번씩 읽는다. 결과 목록을 돌려준다 (로그에도 남긴다)."""
    groups: dict[TransformersConfig, tuple[str, list[str]]] = {}
    for key in PRELOAD_KEYS:
        try:
            client = provider.get(key)
        except LLMError as exc:  # 막힌 엔진·설정 오류: 요청 때도 같은 오류가 나므로 여기서는 건너뛴다
            logger.warning("기동 시 모델 미리 올리기: %s 엔진을 만들지 못해 건너뜁니다 (%s)", key, exc)
            continue
        config = getattr(client, "config", None)
        if not isinstance(config, TransformersConfig):  # FAKE 등 transformers 가 아닌 엔진
            continue
        groups.setdefault(config, (client.name, []))[1].append(key)

    results: list[PreloadResult] = []
    for config, (engine, keys) in groups.items():
        start = time.perf_counter()
        try:
            load(config, engine)
            ok = True
        except LLMError as exc:
            ok = False
            logger.warning(
                "기동 시 모델을 올리지 못했습니다 (서버는 계속 뜹니다. 요청 때 다시 시도): model=%s keys=%s (%s)",
                config.model_id, ",".join(keys), exc,
            )
        seconds = time.perf_counter() - start
        if ok:
            logger.info("기동 시 모델을 올렸습니다: model=%s device=%s keys=%s (%.1f초)",
                        config.model_id, config.device, ",".join(keys), seconds)
        results.append(PreloadResult(config.model_id, tuple(keys), ok, seconds))
    if not groups:
        logger.info("기동 시 미리 올릴 transformers 모델이 없습니다 (모두 FAKE 이거나 설정 오류).")
    return results
