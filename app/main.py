from fastapi import FastAPI
from app.api import health_router
from app.core import get_settings
from app.api.v1.agent.rule_input_extract import router as rule_input_extract_router
from fastapi.exceptions import RequestValidationError
from app.api.v1.agent.agent_run import router as agent_run_router
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

    # 라우터도 create_app() 안에서 붙인다 (테스트의 TestClient(create_app())에도 경로가 있게)
    app.include_router(health_router)
    app.include_router(agent_run_router)
    app.include_router(rule_input_extract_router)
    return app

app = create_app()