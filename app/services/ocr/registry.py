import threading
from app.exceptions.app_errors import AppError
from app.services.ocr.base import OcrEngine
from app.services.ocr.policy import ensure_engine_allowed

class OcrEngineFactory:
    """요청된 엔진만 지연 생성하고 캐시한다. 모델 로딩이 무거워 요청마다 만들지 않는다."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._cache: dict[str, OcrEngine] = {}
        self._lock = threading.Lock()

    def __call__(self, name: str) -> OcrEngine:
        # 서비스에서도 검사하지만, 직접 호출돼도 CLOSED에서는 절대 만들지 않는다
        ensure_engine_allowed(name)

        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._create(name)
            return self._cache[name]

    def _create(self, name: str) -> OcrEngine:
        if name == "TESSERACT":
            from app.services.ocr.tessertact import TesseractEngine            
            return TesseractEngine()

        if name == "PADDLE":
            from app.services.ocr.paddle import PaddleEngine
            return PaddleEngine()

        if name == "OPENAI":
            s = self._settings
            model = s.openai_model
            if s.openai_api_key is None or model is None:
                raise AppError(
                    503,
                    "ENGINE_NOT_CONFIGURED",
                    "OPENAI 엔진 설정(OPENAI_API_KEY, 모델)이 없습니다.",
                )
            from app.services.ocr.openai_ocr import OpenAIOcrEngine
            return OpenAIOcrEngine(
                api_key=s.openai_api_key.get_secret_value(),
                model=model,
                timeout_seconds=s.ai_timeout_seconds,
                max_output_tokens=s.ai_max_output_tokens
            )

        raise AppError(
            code="ENGINE_NOT_SUPPORTED",
            message=f"지원하지 않는 엔진: {name}", 
            status_code=400, 
            )