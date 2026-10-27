from app.schemas.error_schemas import ErrorDetail
"""비즈니스 예외 정의.

서비스와 의존성 계층에서 던지는 도메인 예외 모음. FastAPI를 알지 못하며,
HTTP 응답으로의 변환은 `business_error_handler`가 한 곳에서 담당한다.
"""

class BusinessError(Exception):
    """모든 비즈니스 예외의 기반 클래스.

    Attributes:
        code: 에러 식별자. (예: `FILE_TOO_LARGE`) 응답의 `error.code`가 된다.
        message: 사용자에게 보여줄 수 있는 요약 메시지.
        status_code: 응답에 사용할 HTTP 상태 코드.
        details: 세부 사유 목록. 없으면 None이며, 응답에서는 빈 배열로 변환된다.
    """
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: list[ErrorDetail] | None = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details

        super().__init__(message)