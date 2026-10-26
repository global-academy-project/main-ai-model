from app.core.logging import setup_logging
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from app.core.config import get_settings
from app.exceptions.business_errors import BusinessError
from app.exceptions.app_errors import AppError
from app.handlers.exception_handlers import (
    business_error_handler,
    unexpected_error_handler,
    validation_error_handler,
    app_error_handler
)
from app.api.health import router as health_router
from app.api.v1.ocr import router as ocr_router

setup_logging()
settings = get_settings()

### app을 생성할 때 exception_handler를 등록함
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug
    )

    app.add_exception_handler(
        BusinessError,
        business_error_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )

    app.add_exception_handler(
        AppError,
        app_error_handler,
    )

    return app

app = create_app()

app.include_router(
    health_router
)

app.include_router(
    ocr_router,
    prefix="/api/v1"
)
