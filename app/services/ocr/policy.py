from app.exceptions.business_errors import EngineNotAllowedError
from app.core.config import get_settings

settings = get_settings()

def ensure_engine_allowed(engine: str) -> None:
    """엔진 객체를 만들기 전에 먼저 호출한다."""
    try:
        settings.check_engine(engine)
    except ValueError:
        raise EngineNotAllowedError(selected_engine=engine)