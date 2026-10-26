from app.schemas.error import ErrorDetail
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

class UnsupportedFileTypeError(BusinessError):
    """업로드된 파일이 지원하는 형식이 아닐 때 발생한다. status_code = 400

    Args:
        file_type: 요청에서 확인된 파일 형식.
    """
    def __init__(self, file_type: str):
        super().__init__(
            code="UNSUPPORTED_FILE_TYPE",
            message="지원하지 않는 파일 형식입니다.",
            status_code=400,
            details=[
                ErrorDetail(
                    field=None, 
                    reason=f"파일 형식이 {file_type}가 아닙니다."
                )
            ]
        )

class FileTooLargeError(BusinessError):
    """업로드된 파일이 크기 제한을 넘을 때 발생한다. status_code = 413

    Args:
        max_size: 허용되는 최대 크기(MB).
    """
    def __init__(self, max_size: int):
        super().__init__(
            code="FILE_TOO_LARGE",
            message="파일 크기가 제한을 초과했습니다.",
            status_code=413,
            details=[
                ErrorDetail(
                    field=None, 
                    reason=f"파일 크기가 {max_size}MB보다 큽니다."
                )
           ]
        )

class PageLimitExceededError(BusinessError):
    """PDF 페이지 수가 제한을 넘을 때 발생한다. status_code = 400

    Args:
        page_limit: 허용되는 최대 페이지 수.
    """
    def __init__(self, page_limit: int):
        super().__init__(
            code="PAGE_LIMIT_EXCEEDED",
            message="최대 페이지 수 제한을 초과했습니다.",
            status_code=400,
            details=[
                ErrorDetail(
                    field=None, 
                    reason=f"페이지 수가 {page_limit}보다 큽니다."
                )
            ]
        )

class EngineNotAllowedError(BusinessError):
    def __init__(self, selected_engine: str):
        super().__init__(
            code="ENGINE_NOT_ALLOWED",
            message="현재 사용할 수 없는 엔진입니다.",
            status_code=400,
            details=[
                ErrorDetail(
                    field=None, 
                    reason=f"현재 설정에서 {selected_engine}을 사용할 수 없습니다."
                )
            ]
        )