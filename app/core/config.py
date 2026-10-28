from typing import Literal, get_args
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

"""애플리케이션 설정.

환경 변수와 `.env` 파일에서 값을 읽어 `Settings`로 제공한다.
엔진 선택과 네트워크 모드의 조합은 `Settings` 생성 시점에 검증하므로,
잘못된 조합이면 서버가 기동되지 않는다.
"""

# 사용 가능한 엔진 이름. 엔진을 추가할 때는 external 분류(app.core.engines)도 함께 갱신한다.
Engine = Literal["FAKE", "OPENAI", "CATEGORIZATOR"]
EXTERNAL_ENGINES: frozenset[str] = frozenset({"OPENAI"})

class Settings(BaseSettings):
    """환경 변수 기반 애플리케이션 설정.

    필드 이름은 대소문자를 구분하지 않는 환경 변수 이름과 대응한다.
    (예: `network_env` -> `NETWORK_ENV`)

    Attributes:
        network_env: 네트워크 환경. CLOSED(폐쇄망)에서는 외부 호출 엔진을 쓸 수 없다.
        default_engine: 어떤 시나리오를 사용할지 결정하는 엔진.
        openai_api_key: OpenAI API 키. engine이 OPENAI일 때만 필요하다.
        openai_model: OpenAI 모델 이름. engine이 OPENAI일 때만 필요하다.
        ai_timeout_seconds: AI 호출 제한 시간(초).
        ai_max_output_tokens: AI 응답의 최대 출력 토큰 수.
    """    
    network_env: Literal["OPEN", "CLOSED"]

    openai_api_key: SecretStr | None = None
    openai_model: str | None = None

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

    scenario_engines: dict[str, dict[str, str]] = {
        "default": {"engine": "CATEGORIZATOR", "model_id": "Qwen/Qwen3-0.6B"},
    }

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
    
    def is_engine_blocked(self, engine: Engine) -> bool:
        """폐쇄망에서 외부 엔진을 쓰려는가."""
        return engine in EXTERNAL_ENGINES and self.network_env == "CLOSED"

    def engine_config_problem(self, engine: Engine) -> str | None:
        """필수 설정이 빠졌으면 사유 문자열, 괜찮으면 None."""
        if engine == "OPENAI":
            if self.openai_api_key is None:
                return "OPENAI_API_KEY가 필요합니다."
            if not self.openai_model:
                return "OPENAI_MODEL이 필요합니다."
        return None
    
    def engine_problem(self, engine: Engine) -> str | None:
        """기동 검증용: 막힘과 설정 문제를 합쳐 사유 하나로."""
        if self.is_engine_blocked(engine):
            return f"폐쇄망에서는 {engine}을(를) 사용할 수 없습니다."
        return self.engine_config_problem(engine)

    @model_validator(mode="after")
    def validate_default_engine(self) -> "Settings":
        """기본 엔진이 실행 가능한지 기동 시점에 검증한다."""
        if problem := self.engine_problem("CATEGORIZATOR"):
            raise ValueError(problem)
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