from typing import Literal, get_args
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

"""애플리케이션 설정.

환경 변수와 `.env` 파일에서 값을 읽어 `Settings`로 제공한다.
엔진 선택과 네트워크 모드의 조합은 `Settings` 생성 시점에 검증하므로,
잘못된 조합이면 서버가 기동되지 않는다.
"""
BASE_DIR = Path(__file__).resolve().parents[2]
# 사용 가능한 엔진 이름. 엔진을 추가할 때는 external 분류(app.core.engines)도 함께 갱신한다.
Engine = Literal["AUTO", "PDFIUM", "TESSERACT", "PADDLE", "OPENAI"]
EXTERNAL_ENGINES: frozenset[str] = frozenset({"OPENAI"})

class Settings(BaseSettings):
    """환경 변수 기반 애플리케이션 설정.

    필드 이름은 대소문자를 구분하지 않는 환경 변수 이름과 대응한다.
    (예: `network_env` -> `NETWORK_ENV`)

    Attributes:
        network_env: 네트워크 환경. CLOSED(폐쇄망)에서는 외부 호출 엔진을 쓸 수 없다.
        engine: 기본으로 사용할 엔진.
        openai_api_key: OpenAI API 키. engine이 OPENAI일 때만 필요하다.
        openai_model: OpenAI 모델 이름. engine이 OPENAI일 때만 필요하다.
        ai_timeout_seconds: AI 호출 제한 시간(초).
        ai_max_output_tokens: AI 응답의 최대 출력 토큰 수.
    """
    app_name: str
    app_version: str
    app_env: Literal["development", "test", "production"]
        
    debug: bool = False

    network_env: Literal["OPEN", "CLOSED"]
    engine: Engine

    openai_api_key: SecretStr | None = None
    openai_model: str | None = None

    pdf_backend: Literal["PDFIUM"] = "PDFIUM"
    default_ocr_engine: Literal["TESSERACT", "PADDLE", "OPENAI"] = "OPENAI"

    ai_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
    )
    ai_max_output_tokens: int = Field(
        default=1000,
        ge=1,
        le=100_000,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def available_engines(self) -> list[str]:
        """선언된 모든 엔진 이름을 반환한다."""
        return list(get_args(Engine))
    
    def check_engine(self, engine: Engine) -> None:
        """엔진이 현재 환경에서 실행 가능한지 검증한다.

        Raises:
            ValueError: 폐쇄망에서 외부 엔진을 고르거나, 필수 설정이 없을 때.
        """
        if engine in EXTERNAL_ENGINES and self.network_env == "CLOSED":
            raise ValueError(f"폐쇄망에서는 {engine}을(를) 사용할 수 없습니다.")
        if engine == "OPENAI":
            if self.openai_api_key is None:
                raise ValueError("OPENAI를 쓰려면 OPENAI_API_KEY가 필요합니다.")
            if not self.openai_model:
                raise ValueError("OPENAI를 쓰려면 OPENAI_MODEL이 필요합니다.")
    
    @model_validator(mode="after")
    def validate_default_engine(self) -> "Settings":
        """기본 엔진이 실행 가능한지 기동 시점에 검증한다."""
        self.check_engine(self.engine)
        self.check_engine(self.default_ocr_engine)
        return self
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
    
    # @property
    # def selectable_engines(self) -> list[Engine]:
    #     """현재 환경에서 실제로 선택 가능한 엔진만 반환한다."""
    #     usable: list[Engine] = []
    #     for name in get_args(Engine):
    #         try:
    #             self.check_engine(name)
    #         except ValueError:
    #             continue
    #         usable.append(name)
    #     return usable