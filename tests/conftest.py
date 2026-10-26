"""테스트 공통 설정.

app 모듈은 import 시점에 설정(get_settings)을 읽으므로, 환경 변수를 먼저 고정한다.
실제 .env 가 있어도 여기서 덮어써서 테스트가 외부 API 키나 개발 설정에 영향을 받지 않게 한다.
"""
import os

os.environ.update({
    "APP_NAME": "ocr-test",
    "APP_VERSION": "0.0.0",
    "APP_ENV": "test",
    "DEBUG": "false",
    "NETWORK_ENV": "OPEN",
    "ENGINE": "OPENAI",
    "PDF_BACKEND": "PDFIUM",
    "DEFAULT_OCR_ENGINE": "TESSERACT",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",   # 가짜 값. 실제 호출은 하지 않는다.
    "OPENAI_MODEL": "test-model",
    "AI_TIMEOUT_SECONDS": "30",
    "AI_MAX_OUTPUT_TOKENS": "1000",
})
