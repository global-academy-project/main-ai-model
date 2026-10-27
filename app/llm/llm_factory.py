"""설정에 따라 엔진을 고르는 유일한 지점. 새 엔진은 여기에 등록만 하면 된다."""
from __future__ import annotations
from functools import lru_cache
from app.core import Settings, Engine
from .llm_base import LLMClient
from app.exceptions import LLMConfigError, EngineBlockedError

def build_llm_client(
    *,
    settings: Settings,
    selected_engine: Engine | None = None,
) -> LLMClient:
    engine = selected_engine or settings.default_llm_engine

    if settings.is_engine_blocked(engine):
        raise EngineBlockedError(engine)
    
    if problem := settings.engine_config_problem(engine):
        raise LLMConfigError(problem, engine=engine)

    match engine:
        case "FAKE":
            from .llm_fake import FakeLLMClient
            return FakeLLMClient()
        case "OPENAI":
            # SDK는 지연 임포트
            raise LLMConfigError("OPENAI 엔진은 아직 구현되지 않았습니다.", engine=engine)
        case "CATEGORIZATOR":
            raise LLMConfigError("CATEGORIZATOR 엔진은 아직 구현되지 않았습니다.", engine=engine)
        case _:
            raise LLMConfigError("지원하지 않는 엔진입니다.", engine=engine)

