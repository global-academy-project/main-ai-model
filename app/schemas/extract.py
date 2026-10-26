from pydantic import BaseModel, Field
from enum import StrEnum
from pydantic import field_validator
from pydantic import field_serializer
from app.schemas.base import CamelModel
from app.core.config import Engine

class PageSource(StrEnum):
    TEXT_LAYER = "TEXT_LAYER"
    OCR = "OCR"

class PdfInput(BaseModel):
    """검증을 통과한 PDF 업로드 데이터.

    get_pdf_input 의존성에서 content-type, 크기, %PDF 시그니처 검증을
    마친 뒤 생성되므로, 서비스 계층에서는 올바른 PDF라고 가정해도 됩니다.

    외부 통신용으로는 사용하지 않습니다.

    Attributes:
        filename: 업로드된 원본 파일명. 파일명이 없으면 "unknownFileName".
        content: PDF 파일의 바이트 내용.
        pages: PDF 파일의 페이지 수
    """
    filename: str
    content: bytes
    pages: int

DEFAULT_LANG = ["ko", "en"]

class ExtractOptions(BaseModel):
    engine: Engine = "AUTO"
    extract_images: bool = False
    lang: list[str] = ["ko", "en"]

    @field_validator("lang", mode="before")
    @classmethod
    def _normalize_lang(cls, value):
        """["ko,en"], "ko,en", ["ko", "en"]를 모두 ["ko", "en"]으로 맞춘다."""
        items = [value] if isinstance(value, str) else list(value or [])
        codes: list[str] = []
        for item in items:
            for code in str(item).split(","):
                code = code.strip().lower()
                if code and code not in codes:
                    codes.append(code)
        return codes or list(DEFAULT_LANG)

class PageInfo(CamelModel):
    page_no: int
    source: PageSource
    text : str
    confidence: float | None
    processing_ms: int

    @field_serializer("confidence")
    def _round_confidence(self, value: float | None) -> float | None:
        return None if value is None else round(value, 1)


class ImageInfo(CamelModel):
    page_no: int
    format: str
    width : int
    height: int
    data_base64: str

    @field_validator("format")
    @classmethod
    def _lowercase_format(cls, v: str) -> str:
        return v.lower()

class ExtractData(CamelModel):
    """PDF 요약 결과 데이터

    Attributes:
        engine : 요약에 사용한 엔진.
        networkMode : 서버의 네트워크 환경.
        pageCount : pdf 페이지 수
        totalMs : 전체 처리 시간
        avgConfidence = OCR 처리된 페이지의 평균 신뢰도
        pages = 각 페이지 정보를 담은 배열.
        images = 각 이미지 정보를 담은 배열.
        summary : PDF 내용을 요약한 텍스트.
    """
    
    engine: str = Field(description="요약에 사용한 엔진.")  
    network_mode: str = Field(description="서버의 네트워크 환경.")
    page_count: int = Field(description="pdf 페이지 수.")
    total_ms: float = Field(description="전체 처리 시간.")
    avg_confidence : float | None = Field(description="OCR 처리된 페이지의 평균 신뢰도.")
    pages : list[PageInfo] = Field(default_factory=list, description="각 페이지 정보를 담은 배열")
    images : list[ImageInfo] = Field(default_factory=list, description="각 이미지 정보를 담은 배열.")
    summary: str = Field(default="임시 텍스트 ",description="PDF 내용을 요약한 텍스트.")
    
    @field_serializer("avg_confidence")
    def _round_avg_confidence(self, value: float | None) -> float | None:
        return None if value is None else round(value, 1)