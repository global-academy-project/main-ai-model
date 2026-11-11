from typing import Literal, get_args
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


rag_backend: str | None = "app.rag.tool_backend:build_search_backend"

"""애플리케이션 설정.

환경 변수와 `.env` 파일에서 값을 읽어 `Settings`로 제공한다.
엔진 선택과 네트워크 모드의 조합은 `Settings` 생성 시점에 검증하므로,
잘못된 조합이면 서버가 기동되지 않는다.
"""

# 사용 가능한 엔진 이름. 엔진을 추가할 때는 external 분류(app.core.engines)도 함께 갱신한다.
Engine = Literal["FAKE", "CATEGORIZATOR", "WORK_REGISTER", "CONFINED_SPACE_GATE", "NEAR_MISS_WARNING", "DAILY_SAFETY_LOG", "AUDIT_QUERY", "SMALL_SITE_GUIDE"]
EXTERNAL_ENGINES: frozenset[str] = frozenset({"OPENAI"})

_SCENARIO_KEYS = frozenset({"router", "default", "A", "B", "C", "D", "E", "F"})
_BINDING_FIELDS = frozenset({"engine", "model_id"})

# SCENARIO_MODELS 표 아래, class Settings 위
# 통합 채팅창에서 분류할 시나리오 (글자, 쉼표로). A와 F는 함께 넣는다. D·E는 처리기를 붙일 때 자기 글자를 더한다.
CHAT_SCENARIOS = "A,D,F"

# ── 시나리오별 LLM 모델 선택표 ──────────────────────────────────────────────
# 각 담당자는 자기 시나리오 줄만 고친다. 고친 뒤 서버를 다시 띄우면 적용된다.
#
#   engine   : "FAKE"           가짜 응답 (모델 없음, 개발·테스트용)
#              "CATEGORIZATOR"  transformers 로컬 모델 (처음 쓸 때 읽는다)

#   model_id : CATEGORIZATOR일 때만. 허깅페이스 이름(예: "Qwen/Qwen3-0.6B") 또는 로컬 폴더 경로
#
# - 줄이 없거나 주석이면 "default" 줄을 따른다.
# - 여러 시나리오가 같은 모델을 고르면 메모리에는 한 번만 올라간다.
# - 폐쇄망(NETWORK_ENV=CLOSED)에서는 미리 받아 둔 모델만 쓸 수 있다.
# - 실제로 적용된 값 확인: python scripts/show_scenario_models.py
# - .env 에 SCENARIO_ENGINES 가 있으면 이 표 **전체 대신** 그것을 쓴다. 평소에는 .env 에 두지 않는다.
SCENARIO_MODELS: dict[str, dict[str, str]] = {
    # 담당 · 시나리오
    "default": {"engine": "CATEGORIZATOR", "model_id": "Qwen/Qwen3-0.6B"},  # 아래에서 고르지 않은 것 전부
    "router": {"engine": "CATEGORIZATOR", "model_id": "Qwen/Qwen3-0.6B"},  # B · 시나리오 분류 (채팅 접수)
    "A": {"engine": "WORK_REGISTER", "model_id": "Qwen/Qwen3-0.6B"},  # B · 작업 등록 → 되묻기 → 서류 초안
    "B": {"engine": "CONFINED_SPACE_GATE", "model_id": "Qwen/Qwen3-0.6B"},  # C · 밀폐공간 작업 → 기준 미달 차단 (문장 추출. E와 같은 1.7B를 같이 씀, 어댑터 extract)
    # "C": {"engine": "NEAR_MISS_WARNING", "model_id": "모델 이름"},  # A · 아차사고 보고 → 다음 작업 경고
    "D": {"engine": "DAILY_SAFETY_LOG", "model_id": "Qwen/Qwen3-1.7B"},  # D · 마감 현황·TBM·안전일지 기록 질의
     "E": {"engine": "AUDIT_QUERY", "model_id": "Qwen/Qwen3-0.6B"},  # D · 감사 대응 질의
    # "F": {"engine": "SMALL_SITE_GUIDE", "model_id": "모델 이름"},  # B · 소규모 사업장 안내
}

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

    # 시나리오별 엔진·모델. 기본값은 위 SCENARIO_MODELS 표다 (복사본을 써서 표 자체는 바뀌지 않는다).
    scenario_engines: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {key: dict(binding) for key, binding in SCENARIO_MODELS.items()}
    )

    # LoRA 어댑터 이름 → 로컬 폴더 (env LLM_ADAPTERS, JSON 한 줄). 비우면 어댑터 없이 기본 모델만 쓴다 (기본값).
    #   .env 예) LLM_ADAPTERS={"extract": "/models/adapters/extract", "record_query": "/models/adapters/record_query"}
    #   Windows 경로는 슬래시로 적는다 (예: "D:/models/adapters/extract"). 역슬래시는 JSON 에서 \\ 로 두 번 적어야 한다.
    # - 이름은 엔진 클래스의 adapter 와 짝을 맞춘다 (app/llm/engines/llm_<시나리오>.py 의 "LoRA 어댑터" 주석에 쓰는 순서가 있다).
    # - 어댑터는 머지하지 않고 같은 기본 모델 위에 붙여 쓴다. 학습한 기본 모델(같은 model_id·같은 버전)과 같아야 한다.
    # - 어댑터를 새로 넣거나 바꾸면 서버를 다시 띄운다 (처음 쓸 때 한 번만 읽는다).
    llm_adapters: dict[str, str] = Field(default_factory=dict)

    # 사진 인식 (C, TBM 사진 사람 수). 모델 파일은 로컬 경로만 읽는다 (scripts/download_models.py 로 미리 받아 둔다).
    #   VISION_MODEL_PATH 예) /models/yolox/yolox_nano.onnx. 비우거나 파일이 없으면 /api/v1/vision/* 는 503 (Spring 은 파일 정보 확인만 하고 계속)
    vision_model_path: str | None = None
    vision_input_size: int = Field(default=416, ge=64, le=1280)          # YOLOX-Nano 입력 크기 (공식 416)
    vision_score_threshold: float = Field(default=0.4, gt=0, lt=1)       # 이 점수 미만 사람 박스는 버림
    vision_nms_threshold: float = Field(default=0.45, gt=0, lt=1)        # 이보다 많이 겹치면 점수 높은 박스만 남김
    vision_threads: int | None = Field(default=None, ge=1, le=64)        # ONNX CPU 스레드 수. 비우면 onnxruntime 기본 (1.7B 생성과 CPU 를 나눠 쓸 때 줄인다)

    # 음성 → 글 (C, Whisper small 을 faster-whisper 로). 모델 폴더는 로컬만 읽는다 (scripts/download_models.py --only stt).
    #   STT_MODEL_PATH 예) /models/whisper-small-ko. 비우거나 폴더가 없으면 /api/v1/stt/* 는 503 (Spring 은 "직접 적어 주세요")
    stt_model_path: str | None = None
    # 응답·기록에 남길 모델 이름 (STT_MODEL_PATH 폴더의 모델과 맞춘다). 한국어 모델이면 whisper-small-ko (download_models 출력 그대로)
    stt_model_name: str = Field(default="whisper-small", min_length=1, max_length=100)
    stt_compute_type: Literal["int8", "int8_float32", "float32"] = "int8"   # CPU: int8 이 빠르고 메모리가 적다
    stt_threads: int | None = Field(default=None, ge=1, le=64)              # CPU 스레드 수. 비우면 faster-whisper 기본
    stt_beam_size: int = Field(default=1, ge=1, le=5)                        # 1 이 가장 빠름. 정확도가 부족하면 늘린다
    stt_max_seconds: float = Field(default=35.0, gt=0, le=120)               # 이보다 긴 녹음은 422 (Spring 은 30초에서 끊음)

    # A의 검색 함수 위치 '모듈경로:함수이름' (예: app.rag.tool_backend:build_search_backend).
    # 비우면 mock 검색 도구를 그대로 쓴다. 서버 설정으로만 정한다 (요청자가 고를 수 없다).
    rag_backend: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    chat_scenarios: str = CHAT_SCENARIOS

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
            if engine == "FAKE":
                return f"SCENARIO_ENGINES['{key}']: FAKE 엔진은 model_id를 가질 수 없습니다."
            if not model_id.strip():
                return f"SCENARIO_ENGINES['{key}']의 model_id가 비어 있습니다."
        elif engine != "FAKE" and not getattr(self, "categorizator_model_id", None):
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