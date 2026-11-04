"""CATEGORIZATOR 엔진 (router·분류 전용 클라이언트) · 주인: B

transformers 모델을 분류에 쓰는 클라이언트다. 모델 읽기·생성·오류 처리는 `LLMClient.chat()`(llm_base.py)가
`llm_transformers.py`를 불러 맡으므로 여기서는 설정만 가진다. router 엔진(`llm_router.py`)과 기본 CATEGORIZATOR 엔진이 쓴다.
다른 시나리오 엔진(work_register 등)은 이 클래스가 아니라 `LLMClient`를 상속하고, 각자 모듈에 클라이언트 클래스와 빌드 함수를 가진다.
`router`만 이 클래스를 상속한 `RouterClient`(llm_router.py)를 쓴다.
"""

from __future__ import annotations

from typing import Any

from app.exceptions import LLMConfigError

from ..llm_base import LLMClient
from ..llm_transformers import TransformersConfig, clear_model_cache  # noqa: F401  (clear_model_cache는 테스트가 쓴다)

ENGINE_NAME = "CATEGORIZATOR"
CategorizatorConfig = TransformersConfig


class CategorizatorClient(LLMClient):
    name = ENGINE_NAME

    def __init__(self, config: TransformersConfig) -> None:
        if not config.model_id:
            raise ValueError("model_id가 비어 있습니다.")
        if config.max_new_tokens <= 0:
            raise ValueError("max_new_tokens는 1 이상이어야 합니다.")
        self.config = config


def build_categorizator_client(*, settings: Any, model_id: str | None = None) -> CategorizatorClient:
    """`Settings`에서 설정을 읽어 클라이언트를 만든다 (모델은 첫 호출 때 읽는다).

    `model_id`를 주면 `CATEGORIZATOR_MODEL_ID` 대신 그 모델을 쓴다 (시나리오별 모델).
    `NETWORK_ENV=CLOSED`면 허브 다운로드를 시도하지 않고 로컬 파일만 쓴다.
    """
    model_id = model_id or getattr(settings, "categorizator_model_id", None)
    if not model_id:
        raise LLMConfigError("CATEGORIZATOR_MODEL_ID가 설정되지 않았습니다.", engine=ENGINE_NAME)
    local_only = bool(getattr(settings, "categorizator_local_files_only", False)) or (
        settings.network_env == "CLOSED"
    )
    return CategorizatorClient(
        TransformersConfig(
            model_id=model_id,
            max_new_tokens=settings.ai_max_output_tokens,  # 모든 엔진이 같은 값 (config.py의 ai_max_output_tokens)
            device=getattr(settings, "categorizator_device", "cpu"),
            local_files_only=local_only,
            revision=getattr(settings, "categorizator_revision", None),
            enable_thinking=getattr(settings, "categorizator_enable_thinking", False),
        )
    )