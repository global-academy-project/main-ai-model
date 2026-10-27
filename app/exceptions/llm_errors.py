class LLMError(Exception):
    """엔진 호출 실패(타임아웃, 인증, 형식 오류 등).

    Attributes:
        engine: 문제가 된 엔진 종류. (예: `anthropic`) 모르면 None.
    """

    def __init__(self, message: str = "", *, engine: str | None = None):
        self.engine = engine
        super().__init__(message)


class LLMConfigError(LLMError):
    """엔진 설정 오류(알 수 없는 엔진, 필수 환경 변수 누락 등)."""

    def __init__(self, message: str, *, engine: str | None = None):
        super().__init__(message, engine=engine)


class EngineBlockedError(LLMError):
    """NETWORK_ENV=CLOSED 에서 외부 엔진을 쓰려 할 때."""

    def __init__(self, engine: str):
        super().__init__(
            f"엔진 '{engine}'은(는) NETWORK_ENV=CLOSED에서 사용할 수 없습니다.",
            engine=engine,
        )