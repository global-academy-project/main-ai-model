from pydantic import BaseModel, Field
from app.schemas.base import CamelModel

class EngineInfo(BaseModel):
    """엔진 하나의 상태.

    Attributes:
        name: 엔진 이름.
        available: 현재 사용 가능 여부.
        external: 외부 네트워크 호출 여부.
        reason: 사용 불가 사유. 사용 가능하면 None.
    """

    name: str = Field(description="엔진 이름.")
    available: bool = Field(description="현재 사용 가능 여부.")
    external: bool = Field(description="외부 네트워크 호출 여부..")
    reason: str | None = Field(default=None, description="외부 네트워크 호출 여부..")

class EnginesData(CamelModel):
    """엔진 목록 조회 결과 데이터.
            
    Attributes:
        network_mode : 서버의 네트워크 환경.
        default_engine : 기본값으로 지정된 엔진.
        engines : 가용한 엔진 목록.
    """

    network_mode : str = Field(description="서버의 네트워크 환경.")
    default_engine : str = Field(description="기본값으로 지정된 엔진")
    engines : list[EngineInfo] = Field(description="가용한 엔진 상태 목록.")