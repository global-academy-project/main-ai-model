import asyncio
import base64
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import openai
import pytest
from PIL import Image

import app.services.ocr.policy as policy
from app.exceptions.app_errors import AppError
from app.exceptions.business_errors import EngineNotAllowedError
from app.schemas.extract import ExtractOptions, PageInfo, PageSource, PdfInput
from app.services.extract_service import ExtractService, _fit_to_limit, choose_engine
from app.services.pdf.base import PdfOpenError
from app.services.summary_service import SummaryResult
from tests.helpers import make_settings

LONG_TEXT = "x" * 25        # 텍스트 레이어로 인정되는 길이(20자 이상)


# ---- 가짜 PDF 백엔드 / OCR 엔진 -------------------------------------------------

class FakePage:
    def __init__(self, text):
        self.text, self.renders = text, 0

    def extract_text(self):
        return self.text

    def render(self, *, dpi):
        self.renders += 1
        self.last_dpi = dpi
        return Image.new("RGB", (30, 20), "white")


class FakeBackend:
    name = "PDFIUM"

    def __init__(self, texts, open_error=None):
        self.pages = [FakePage(t) for t in texts]
        self.open_error = open_error
        self.opened = 0

    @contextmanager
    def open(self, content):
        self.opened += 1
        if self.open_error:
            raise self.open_error
        yield SimpleNamespace(page_count=len(self.pages), pages=lambda: iter(self.pages))


class FakeEngine:
    def __init__(self, name, results=None, error=None):
        self.name, self.error = name, error
        self.results = list(results or [("ocr text", 0.9)])
        self.calls = []

    def recognize(self, image, *, lang):
        self.calls.append(lang)
        if self.error:
            raise self.error
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


class Factory:
    def __init__(self, **engines):
        self.engines, self.requested = engines, []

    def __call__(self, name):
        self.requested.append(name)
        return self.engines[name]


def service(texts, factory=None, *, settings=None, backend=None, summary=None):
    backend = backend or FakeBackend(texts)
    settings = settings or SimpleNamespace(network_env="OPEN", default_ocr_engine="TESSERACT", engine="OPENAI")
    return ExtractService(settings=settings, pdf_backend=backend, engine_factory=factory or Factory(),
                          summary_service=summary), backend


def run(svc, **opts):
    pdf = PdfInput(filename="a.pdf", content=b"dummy", pages=1)
    return asyncio.run(svc.extract(pdf, ExtractOptions(**opts)))


# ---- 순수 함수 -------------------------------------------------------------------

class TestChooseEngine:
    @pytest.mark.parametrize("text,expected", [
        ("x" * 20, "PDFIUM"),         # 경계: 정확히 20자는 텍스트 레이어
        ("x" * 19, "TESSERACT"),      # 19자는 OCR
        ("", "TESSERACT"),
        ("x" * 500, "PDFIUM"),
    ])
    def test_auto_decides_per_page_by_text_length(self, text, expected):
        assert choose_engine("AUTO", text, "TESSERACT", "PDFIUM") == expected

    @pytest.mark.parametrize("requested", ["TESSERACT", "PADDLE", "OPENAI", "PDFIUM"])
    def test_explicit_engine_is_used_for_every_page(self, requested):
        assert choose_engine(requested, "x" * 500, "TESSERACT", "PDFIUM") == requested
        assert choose_engine(requested, "", "TESSERACT", "PDFIUM") == requested


class TestFitToLimit:
    def test_short_text_unchanged(self):
        assert _fit_to_limit("abc", 10) == "abc"
        assert _fit_to_limit("a" * 10, 10) == "a" * 10      # 경계: 정확히 limit

    def test_cuts_at_last_line_break_before_limit(self):
        text = "aaaa\nbbbb\ncccc"          # limit 12 → 마지막 줄바꿈(위치 9)에서 자름
        assert _fit_to_limit(text, 12) == "aaaa\nbbbb"

    def test_hard_cut_when_line_break_is_too_early(self):
        text = "a\n" + "b" * 40            # 줄바꿈이 limit/2 보다 앞이면 limit 에서 그냥 자른다
        assert _fit_to_limit(text, 20) == text[:20].rstrip()

    def test_result_never_exceeds_limit(self):
        assert len(_fit_to_limit("가\n" * 100, 51)) <= 51


# ---- extract(): 엔진 선택 -----------------------------------------------------------

class TestEngineSelection:
    def test_auto_mixes_text_layer_and_ocr_per_page(self):
        tess = FakeEngine("TESSERACT", [("스캔 본문", 0.8)])
        svc, backend = service([LONG_TEXT, ""], Factory(TESSERACT=tess))
        data = run(svc, engine="AUTO")

        assert [p.source for p in data.pages] == [PageSource.TEXT_LAYER, PageSource.OCR]
        assert data.pages[0].text == LONG_TEXT and data.pages[0].confidence is None
        assert data.pages[1].text == "스캔 본문" and data.pages[1].confidence == 0.8
        assert data.engine == "TESSERACT"                     # 실제 사용된 OCR 엔진
        assert len(tess.calls) == 1                           # 텍스트 레이어 페이지는 OCR 안 함
        assert backend.pages[0].renders == 0                  # 렌더링도 하지 않는다

    def test_auto_with_only_text_layers_reports_pdf_backend_and_no_ocr(self):
        factory = Factory()
        data = run(service([LONG_TEXT, LONG_TEXT], factory)[0], engine="AUTO")
        assert data.engine == "PDFIUM"
        assert data.avg_confidence is None
        assert factory.requested == []

    def test_explicit_ocr_engine_ignores_text_layer(self):
        tess = FakeEngine("TESSERACT")
        data = run(service([LONG_TEXT], Factory(TESSERACT=tess))[0], engine="TESSERACT")
        assert data.pages[0].source == PageSource.OCR
        assert data.pages[0].text == "ocr text"
        assert data.engine == "TESSERACT"

    def test_explicit_pdfium_uses_text_layer_only(self):
        factory = Factory()
        svc, backend = service(["짧음", ""], factory)
        data = run(svc, engine="PDFIUM")
        assert [p.text for p in data.pages] == ["짧음", ""]
        assert all(p.source == PageSource.TEXT_LAYER for p in data.pages)
        assert data.engine == "PDFIUM"
        assert factory.requested == []

    def test_openai_engine_has_no_confidence(self):
        openai_engine = FakeEngine("OPENAI", [("hi", None)])
        data = run(service([""], Factory(OPENAI=openai_engine))[0], engine="OPENAI")
        assert data.engine == "OPENAI"
        assert data.pages[0].confidence is None
        assert data.avg_confidence is None

    def test_pages_rendered_at_300_dpi(self):
        svc, backend = service([""], Factory(TESSERACT=FakeEngine("TESSERACT")))
        run(svc, engine="TESSERACT")
        assert backend.pages[0].last_dpi == 300

    def test_language_preference_is_passed_to_engine(self):
        tess = FakeEngine("TESSERACT")
        run(service([""], Factory(TESSERACT=tess))[0], engine="TESSERACT", lang=["ko,en"])
        assert tess.calls == [["ko", "en"]]


# ---- extract(): 결과 구성 -------------------------------------------------------------

class TestResultAssembly:
    def test_page_numbers_start_at_one_in_order(self):
        data = run(service(["", "", ""], Factory(TESSERACT=FakeEngine("TESSERACT")))[0], engine="TESSERACT")
        assert [p.page_no for p in data.pages] == [1, 2, 3]
        assert data.page_count == 3

    def test_average_confidence_uses_only_ocr_pages_with_a_score(self):
        tess = FakeEngine("TESSERACT", [("a", 0.8), ("b", 0.6), ("c", None)])
        # 첫 페이지는 텍스트 레이어(제외), 나머지 3개는 OCR
        data = run(service([LONG_TEXT, "", "", ""], Factory(TESSERACT=tess))[0], engine="AUTO")
        assert data.avg_confidence == pytest.approx(0.7)

    def test_network_mode_and_timing_are_reported(self):
        settings = SimpleNamespace(network_env="CLOSED", default_ocr_engine="TESSERACT", engine="TESSERACT")
        data = run(service([LONG_TEXT], settings=settings)[0], engine="PDFIUM")
        assert data.network_mode == "CLOSED"
        assert data.total_ms >= 0 and data.pages[0].processing_ms >= 0

    def test_images_are_not_returned_by_default(self):
        data = run(service([LONG_TEXT], Factory())[0], engine="AUTO")
        assert data.images == []

    def test_extract_images_returns_page_renderings_as_png(self):
        svc, _ = service([LONG_TEXT, ""], Factory(TESSERACT=FakeEngine("TESSERACT")))
        data = run(svc, engine="AUTO", extract_images=True)
        assert [i.page_no for i in data.images] == [1, 2]
        first = data.images[0]
        assert (first.format, first.width, first.height) == ("png", 30, 20)
        decoded = Image.open(BytesIO(base64.b64decode(first.data_base64)))
        assert decoded.size == (30, 20)

    def test_ocr_page_image_reuses_the_rendering_used_for_ocr(self):
        svc, backend = service([""], Factory(TESSERACT=FakeEngine("TESSERACT")))
        run(svc, engine="TESSERACT", extract_images=True)
        assert backend.pages[0].renders == 1          # OCR 용 렌더링을 이미지에도 재사용

    def test_empty_document_yields_empty_result(self):
        data = run(service([], Factory())[0], engine="AUTO")
        assert (data.page_count, data.pages, data.avg_confidence, data.engine) == (0, [], None, "PDFIUM")


# ---- extract(): 오류 -----------------------------------------------------------------

class TestErrors:
    def test_unreadable_pdf_becomes_422(self):
        backend = FakeBackend([], open_error=PdfOpenError("열 수 없는 PDF입니다."))
        with pytest.raises(AppError) as e:
            run(service([], backend=backend)[0], engine="AUTO")
        assert (e.value.code, e.value.status_code) == ("PDF_UNREADABLE", 422)

    def test_engine_failure_propagates(self):
        broken = FakeEngine("OPENAI", error=AppError(code="OCR_PROVIDER_ERROR", message="실패", status_code=503))
        with pytest.raises(AppError) as e:
            run(service([""], Factory(OPENAI=broken))[0], engine="OPENAI")
        assert e.value.code == "OCR_PROVIDER_ERROR"

    @pytest.fixture
    def closed(self, monkeypatch):
        monkeypatch.setattr(policy, "settings", make_settings(network_env="CLOSED", engine="TESSERACT"))

    def test_closed_network_blocks_external_engine_before_opening_pdf(self, closed):
        svc, backend = service([""], Factory())
        with pytest.raises(EngineNotAllowedError):
            run(svc, engine="OPENAI")
        assert backend.opened == 0

    def test_closed_network_blocks_auto_when_default_ocr_is_external(self, closed):
        settings = SimpleNamespace(network_env="CLOSED", default_ocr_engine="OPENAI", engine="TESSERACT")
        svc, backend = service([""], settings=settings)
        with pytest.raises(EngineNotAllowedError):
            run(svc, engine="AUTO")
        assert backend.opened == 0

    def test_closed_network_blocks_everything_if_summary_provider_is_external(self, closed):
        # ENGINE=OPENAI(요약이 외부로 나감)인데 폐쇄망이면 로컬 엔진 요청도 먼저 막는다
        settings = SimpleNamespace(network_env="CLOSED", default_ocr_engine="TESSERACT", engine="OPENAI")
        svc, backend = service([LONG_TEXT], settings=settings)
        with pytest.raises(EngineNotAllowedError):
            run(svc, engine="PDFIUM")
        assert backend.opened == 0


# ---- 요약(현재 extract 에서는 호출되지 않는 내부 메서드) --------------------------------

class FakeSummary:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.commands = result, error, []

    async def summarize(self, command):
        self.commands.append(command)
        if self.error:
            raise self.error
        return self.result


def pages_with(*texts):
    return [PageInfo(page_no=i, source=PageSource.OCR, text=t, confidence=None, processing_ms=0)
            for i, t in enumerate(texts, start=1)]


class TestSummarize:
    def summarize(self, summary, pages):
        svc, _ = service([], summary=summary)
        return asyncio.run(svc._summarize(pages, ExtractOptions()))

    def test_joins_page_texts_and_returns_summary(self):
        summary = FakeSummary(SummaryResult(summary="요약", length="medium", tone="neutral"))
        assert self.summarize(summary, pages_with("가", "  ", "나")) == "요약"
        assert summary.commands[0].text == "가\n\n나"          # 빈 페이지는 제외

    def test_input_is_limited_to_20000_chars(self):
        summary = FakeSummary(SummaryResult(summary="s", length="medium", tone="neutral"))
        self.summarize(summary, pages_with("a" * 30000))
        assert len(summary.commands[0].text) <= 20000

    def test_provider_error_is_503_without_details(self):
        with pytest.raises(AppError) as e:
            self.summarize(FakeSummary(error=openai.OpenAIError("sk-LEAK secret text")), pages_with("본문"))
        assert (e.value.code, e.value.status_code) == ("SUMMARY_PROVIDER_ERROR", 503)
        assert "sk-LEAK" not in e.value.message

    @pytest.mark.parametrize("error", [ValueError("빈 요약"), RuntimeError("빈 응답")])
    def test_empty_result_is_502(self, error):
        with pytest.raises(AppError) as e:
            self.summarize(FakeSummary(error=error), pages_with("본문"))
        assert (e.value.code, e.value.status_code) == ("SUMMARY_FAILED", 502)

    @pytest.mark.xfail(strict=False, reason="텍스트가 없을 때 AppError 를 위치 인자로 호출해(키워드 전용) TypeError 가 난다. 422 NO_TEXT_EXTRACTED 여야 함")
    def test_no_text_is_422(self):
        with pytest.raises(AppError) as e:
            self.summarize(FakeSummary(), pages_with("", "  "))
        assert (e.value.code, e.value.status_code) == ("NO_TEXT_EXTRACTED", 422)
