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

_SCENARIO_KEYS = frozenset({"router", "default", "A", "B", "C", "D", "E", "F"})
_BINDING_FIELDS = frozenset({"engine", "model_id"})

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

    check_rules_tool: Literal["MOCK", "SPRING"]
    internal_api_base_url: str | None = None
    internal_api_token: SecretStr | None = None

    scenario_engines: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "default": {"engine": "CATEGORIZATOR", "model_id": "Qwen/Qwen3-0.6B"},
        }
    )

    # A의 검색 함수 위치 '모듈경로:함수이름' (예: app.rag.tool_backend:build_search_backend).
    # 비우면 mock 검색 도구를 그대로 쓴다. 서버 설정으로만 정한다 (요청자가 고를 수 없다).
    rag_backend: str | None = None

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

    def scenario_binding_problem(self, key: str, binding: dict[str, str]) -> str | None:
        """SCENARIO_ENGINES 항목 하나의 문제. 없으면 None."""
        if key not in _SCENARIO_KEYS:
            return f"SCENARIO_ENGINES의 키 '{key}'를 알 수 없습니다."
        if unknown := binding.keys() - _BINDING_FIELDS:
            return f"SCENARIO_ENGINES['{key}']에 알 수 없는 항목이 있습니다: {sorted(unknown)}"
        engine = binding.get("engine")
        if engine not in get_args(Engine):
            return f"SCENARIO_ENGINES['{key}']의 엔진 '{engine}'은 지원하지 않습니다."
        model_id = binding.get("model_id")
        if model_id is not None:
            if engine != "CATEGORIZATOR":
                return f"SCENARIO_ENGINES['{key}']: model_id는 CATEGORIZATOR 엔진만 받습니다."
            if not model_id.strip():
                return f"SCENARIO_ENGINES['{key}']의 model_id가 비어 있습니다."
        elif engine == "CATEGORIZATOR" and not self.categorizator_model_id:
            return f"SCENARIO_ENGINES['{key}']: model_id 또는 CATEGORIZATOR_MODEL_ID가 필요합니다."
        return self.engine_problem(engine)  # 막힌 엔진, 설정 문제는 기존 규칙 그대로
    
    @model_validator(mode="after")
    def validate_default_engine(self) -> "Settings":
        """기본 엔진이 실행 가능한지 기동 시점에 검증한다."""
        for key, binding in self.scenario_engines.items():
            if problem := self.scenario_binding_problem(key, binding):
                raise ValueError(problem)
        return self

    @model_validator(mode="after")
    def validate_internal_api(self) -> "Settings":
        """agent를 spring에서 규칙 확인할 수 있는지 기동 시점에 검증한다."""
        if self.check_rules_tool == "SPRING":
            if self.internal_api_base_url is None:
                raise ValueError("internal_api_base_url이 필요합니다.")
            if self.internal_api_token is None:
                raise ValueError("internal_api_token 필요합니다.")
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