from fastapi import FastAPI
from app.api import health_router
from app.exceptions import BusinessError, LLMError
from app.handlers import (
    business_error_handler,
    llm_error_handler
)

def create_app() -> FastAPI:
    app = FastAPI(
    )

    app.add_exception_handler(
        BusinessError,
        business_error_handler,
    )

    app.add_exception_handler(
        LLMError,
        llm_error_handler,
    )
    return app

app = create_app()

app.include_router(
    health_router
)
