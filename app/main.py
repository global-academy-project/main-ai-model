from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
import logging
from app.agent.scenario.scenario_handlers import load_chat_scenarios
from app.api import health_router
from app.api.v1.agent.agent_run import router as agent_run_router
from app.api.v1.agent.record_query import router as record_query_router
from app.api.v1.agent.rule_input_extract import router as rule_input_extract_router
from app.core import get_settings
from app.exceptions import AgentRunError, BusinessError, LLMError
from app.handlers import (
    business_error_handler,
    llm_error_handler
)
from app.handlers.agent_run_handlers import (
    agent_run_error_handler,
    unhandled_error_handler,
    validation_error_handler,
)
from app.llm.llm_provider import LLMClientProvider

logger = logging.getLogger(__name__)


def _optional_router(path: str):
    """파일(multipart)을 받는 경로 (사진 인식·STT, C). python-multipart 가 없으면 붙이지 않고 경고만 남긴다
    (다른 경로는 그대로 뜨게. Spring 은 이 경로가 실패하면 그 기능 없이 계속한다)."""
    import importlib

    try:
        return importlib.import_module(path).router
    except (ImportError, RuntimeError) as exc:
        logger.warning("%s 경로를 붙이지 않았습니다 (python-multipart 설치 확인): %s", path, exc)
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI()

    app.add_exception_handler(
        BusinessError,
        business_error_handler,
    )

    app.add_exception_handler(
        LLMError,
        llm_error_handler,
    )
    app.add_exception_handler(AgentRunError, agent_run_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    app.state.llm_provider = LLMClientProvider.from_settings(settings)
    # 채팅 범위는 기동 때 한 번 검사한다 (잘못된 값이면 여기서 ValueError로 뜨지 않는다. 요청마다 500이 되지 않게)
    app.state.chat_scenarios = load_chat_scenarios(settings.chat_scenarios)

    # 라우터도 create_app() 안에서 붙인다 (테스트의 TestClient(create_app())에도 경로가 있게)
    app.include_router(health_router)
    app.include_router(agent_run_router)
    app.include_router(record_query_router)
    app.include_router(rule_input_extract_router)
    vision_router = _optional_router("app.api.v1.agent.vision")
    if vision_router is not None:
        app.include_router(vision_router)
    stt_router = _optional_router("app.api.v1.agent.stt")
    if stt_router is not None:
        app.include_router(stt_router)
    return app

app = create_app()