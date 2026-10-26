from typing import Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

T = TypeVar("T")

class CamelModel(BaseModel):
    """파이썬 내부 snake_case와 외부의 camelCase간 호환이 되도록 설정한 BaseModel"""
    # snake_case와 camelCase간 호환을 위한 설정
    model_config = ConfigDict(
        alias_generator=to_camel, 
        populate_by_name=True
        )

class BaseResponse(BaseModel):
    """모든 JSON 응답의 공통 베이스.
    Attributes:
        success: 요청 성공 여부"""
    success: bool

class DataResponse(BaseResponse, Generic[T]):
    """요청된 작업을 성공했을 때, 응답으로 사용하는 DTO

    Attributes:
        success: 요청 성공 여부
        data: 실제 응답 데이터
    """
    success: bool = Field(default=True, description="요청된 작업을 성공적으로 응답함.") 
    data: T = Field(description="실제 응답 데이터.")