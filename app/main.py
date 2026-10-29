from fastapi import FastAPI
from app.api import health_router, chat_router
from app.core import get_settings
from app.exceptions import BusinessError, LLMError
from app.handlers import (
    business_error_handler,
    llm_error_handler
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
    app.state.llm_provider = LLMClientProvider.from_settings(settings)
    return app

app = create_app()

app.include_router(health_router)
app.include_router(chat_router)