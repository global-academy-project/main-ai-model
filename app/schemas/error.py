from pydantic import BaseModel
from app.schemas.base import BaseResponse

class ErrorDetail(BaseModel):
    """에러의 세부 사유 하나.

    Attributes:
        field: 문제가 된 요청 필드 이름. 필드와 무관한 에러면 None.
        reason: 사람이 읽을 수 있는 사유.
    """
    field: str | None = None   # 필드와 무관한 에러면 None
    reason: str

class ErrorBody(BaseModel):
    """에러 내용.

    Attributes:
        code: 프론트가 분기에 사용하는 에러 식별자. (예: `VALIDATION_ERROR`)
        message: 사용자에게 보여줄 수 있는 요약 메시지.
        details: 세부 사유 목록. 없으면 null이 아니라 빈 배열로 내려간다.
    """
    code: str
    message: str
    details: list[ErrorDetail] = []

class ErrorResponse(BaseResponse):
    """에러 응답의 최상위 봉투.

    Attributes:
        success: 항상 False.
        error: 에러 내용.
    """
    success: bool = False
    error: ErrorBody