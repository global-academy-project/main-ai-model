import threading
from contextlib import contextmanager
from typing import Iterator
import pypdfium2 as pdfium
from app.services.pdf.base import PdfOpenError

# PDFium은 스레드 안전하지 않다고 알려져 있어(사용 버전의 pypdfium2 문서에서 확인 권장)
# pdfium 호출은 전역 락으로 직렬화한다. OCR은 락 밖에서 돈다.
_LOCK = threading.RLock()

class _Page:
    def __init__(self, page) -> None:
        self._page = page

    def extract_text(self) -> str:
        with _LOCK:
            textpage = self._page.get_textpage()
            try:
                return textpage.get_text_range().strip()
            finally:
                textpage.close()

    def render(self, *, dpi: int):
        with _LOCK:
            return self._page.render(scale=dpi / 72).to_pil().convert("RGB")


class _Document:
    def __init__(self, doc) -> None:
        self._doc = doc

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def pages(self) -> Iterator[_Page]:
        for index in range(len(self._doc)):
            with _LOCK:
                page = self._doc[index]
            try:
                yield _Page(page)
            finally:
                with _LOCK:
                    page.close()


class PdfiumBackend:
    name = "PDFIUM"

    @contextmanager
    def open(self, content: bytes):
        with _LOCK:
            try:
                doc = pdfium.PdfDocument(content)
            except pdfium.PdfiumError as exc:
                raise PdfOpenError("열 수 없는 PDF입니다.") from exc
        try:
            yield _Document(doc)
        finally:
            with _LOCK:
                doc.close()