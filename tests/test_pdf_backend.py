import pytest
from PIL import Image

from app.exceptions.app_errors import AppError
from app.services.pdf.base import PdfOpenError
from app.services.pdf.pdfium_backend import PdfiumBackend
from app.services.pdf.registry import create_pdf_backend
from tests.helpers import TEXT_LAYER_SENTENCE, blank_pdf, text_pdf


@pytest.fixture
def backend():
    return PdfiumBackend()


def on_first_page(doc, action):
    """첫 페이지에서 action 을 실행한다.

    pages() 는 다음 페이지로 넘어가거나 제너레이터가 닫힐 때 직전 페이지를 닫는다.
    next(iter(doc.pages())) 처럼 제너레이터를 바로 버리면 페이지가 먼저 닫혀 버리므로,
    for 루프 안에서 사용하고 결과만 돌려준다.
    """
    for page in doc.pages():
        return action(page)


def test_backend_name_is_reported_in_responses(backend):
    assert backend.name == "PDFIUM"


def test_page_count_and_order(backend):
    with backend.open(blank_pdf(3)) as doc:
        assert doc.page_count == 3
        assert len(list(doc.pages())) == 3


def test_text_layer_is_extracted_and_stripped(backend):
    with backend.open(text_pdf()) as doc:
        assert on_first_page(doc, lambda p: p.extract_text()) == TEXT_LAYER_SENTENCE


def test_page_without_text_layer_returns_empty_string(backend):
    with backend.open(blank_pdf(1)) as doc:
        assert on_first_page(doc, lambda p: p.extract_text()) == ""


def test_render_returns_rgb_image_scaled_by_dpi(backend):
    with backend.open(text_pdf()) as doc:       # 페이지 크기 300x200pt
        low, high = on_first_page(doc, lambda p: (p.render(dpi=72), p.render(dpi=144)))
    assert isinstance(low, Image.Image) and low.mode == "RGB"
    assert 299 <= low.width <= 301 and 199 <= low.height <= 201
    assert abs(high.width - low.width * 2) <= 2


@pytest.mark.parametrize("content", [b"", b"not a pdf", b"%PDF-1.4 garbage"])
def test_unreadable_pdf_raises_pdf_open_error(backend, content):
    with pytest.raises(PdfOpenError):
        with backend.open(content):
            pass


def test_registry_returns_cached_pdfium_backend():
    a, b = create_pdf_backend("PDFIUM"), create_pdf_backend("PDFIUM")
    assert a.name == "PDFIUM"
    assert a is b


def test_registry_rejects_unknown_backend():
    with pytest.raises(AppError) as e:
        create_pdf_backend("PYMUPDF")
    assert e.value.code == "PDF_BACKEND_NOT_SUPPORTED"
    assert e.value.status_code == 500
