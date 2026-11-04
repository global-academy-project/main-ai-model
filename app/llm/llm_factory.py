"""설정에 따라 엔진을 고르는 유일한 지점. 새 엔진은 여기에 등록만 하면 된다."""
from app.core import Settings, Engine
from .engine_names import MODEL_ID_ENGINES, canonical_engine
from .llm_base import LLMClient
from app.exceptions import LLMConfigError, EngineBlockedError

def build_llm_client(
    *,
    settings: Settings,
    selected_engine: Engine | None = None,
    model_id: str | None = None,
) -> LLMClient:
    """model_id: 시나리오별 모델 지정. transformers 엔진(CATEGORIZATOR·시나리오 엔진)에서만 쓸 수 있다."""
    engine = canonical_engine(selected_engine or settings.default_llm_engine)  # 대문자 시나리오 엔진 이름도 받는다

    if settings.is_engine_blocked(engine):
        raise EngineBlockedError(engine)

    if problem := settings.engine_config_problem(engine):
        raise LLMConfigError(problem, engine=engine)

    if model_id is not None and engine not in MODEL_ID_ENGINES:
        raise LLMConfigError("이 엔진은 모델 지정을 지원하지 않습니다.", engine=engine)

    match engine:
        case "FAKE":
            from .llm_fake import FakeLLMClient
            return FakeLLMClient()
        case "OPENAI":
            # SDK는 지연 임포트
            raise LLMConfigError("OPENAI 엔진은 아직 구현되지 않았습니다.", engine=engine)
        case name if name in MODEL_ID_ENGINES and name != "CATEGORIZATOR":
            # 시나리오 엔진: 엔진 이름 = 시나리오 키, 모듈은 app/llm/engines/llm_<키>.py (각 주인이 고친다)
            from .engines import build_scenario_engine
            return build_scenario_engine(name, settings=settings, model_id=model_id)
        case "CATEGORIZATOR":
            # transformers는 지연 임포트 (이 엔진을 실제로 쓸 때만)
            from .engines.llm_categorizator import build_categorizator_client
            return build_categorizator_client(settings=settings, model_id=model_id)
        case _:
            raise LLMConfigError("지원하지 않는 엔진입니다.", engine=engine)



