from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Iterator, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image


class PdfOpenError(Exception):
    """열 수 없는 PDF(손상, 암호). HTTP 변환은 상위 계층에서 한다."""


class PdfPage(Protocol):
    def extract_text(self) -> str:
        """텍스트 레이어를 strip한 문자열로 반환."""
        ...

    def render(self, *, dpi: int) -> "Image":
        """페이지를 RGB PIL 이미지로 렌더링."""
        ...


class PdfDocument(Protocol):
    @property
    def page_count(self) -> int: ...

    def pages(self) -> Iterator[PdfPage]: ...


class PdfBackend(Protocol):
    name: str  # 응답의 engine 값으로 쓰인다 ("PDFIUM", "PYMUPDF")

    def open(self, content: bytes) -> AbstractContextManager[PdfDocument]: ...