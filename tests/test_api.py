"""HTTP 계층 테스트: 라우팅, 요청 파싱, 검증 오류 응답, 응답 형식(camelCase 봉투)."""
import pytest

pytest.importorskip("httpx")   # TestClient 에 필요

from fastapi.testclient import TestClient

from app.core.dependecies.dependencies import get_extract_service
from app.main import app
from app.schemas.extract import ExtractData
from tests.helpers import TEXT_LAYER_SENTENCE, blank_pdf, text_pdf

URL = "/api/v1/ocr/extract"
client = TestClient(app, raise_server_exceptions=False)


class StubExtractService:
    """실제 OCR 없이 요청 파싱 결과(options)만 확인하기 위한 대역"""

    def __init__(self, error=None):
        self.calls, self.error = [], error

    async def extract(self, pdf, options):
        self.calls.append((pdf, options))
        if self.error:
            raise self.error
        return ExtractData(engine="TESSERACT", network_mode="OPEN", page_count=pdf.pages, total_ms=12.0,
                           avg_confidence=None)


@pytest.fixture
def stub():
    s = StubExtractService()
    app.dependency_overrides[get_extract_service] = lambda: s
    yield s
    app.dependency_overrides.clear()


def post(data=None, *, pdf=None, filename="a.pdf"):
    files = {"file": (filename, pdf if pdf is not None else blank_pdf(2), "application/pdf")}
    return client.post(URL, files=files, data=data or {})


# ---- health / engines ----------------------------------------------------------

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"success": True, "data": {"status": "ok"}}


def test_engines_response_shape():
    r = client.get("/api/v1/ocr/engines")
    body = r.json()
    assert r.status_code == 200 and body["success"] is True
    data = body["data"]
    assert data["networkMode"] == "OPEN"
    assert data["defaultEngine"] == "OPENAI"
    assert [e["name"] for e in data["engines"]] == ["AUTO", "PDFIUM", "TESSERACT", "PADDLE", "OPENAI"]
    openai = data["engines"][-1]
    assert openai["external"] is True and openai["available"] is True


# ---- 요청 파싱 -------------------------------------------------------------------

def test_form_fields_are_parsed_into_options(stub):
    r = post({"engine": "TESSERACT", "extract_images": "true", "lang": "ko,en"})
    assert r.status_code == 200
    pdf, options = stub.calls[0]
    assert pdf.pages == 2 and pdf.filename == "a.pdf"
    assert (options.engine, options.extract_images, options.lang) == ("TESSERACT", True, ["ko", "en"])


def test_defaults_when_form_fields_omitted(stub):
    post()
    _, options = stub.calls[0]
    assert options.engine == "OPENAI"            # 서버 기본 엔진(ENGINE)
    assert options.extract_images is False
    assert options.lang == ["ko", "en"]


def test_repeated_lang_fields_are_accepted(stub):
    client.post(URL, files={"file": ("a.pdf", blank_pdf(1), "application/pdf")},
                data={"lang": ["en", "ko"]})
    assert stub.calls[0][1].lang == ["en", "ko"]


def test_success_response_uses_camel_case_envelope(stub):
    body = post({"engine": "TESSERACT"}).json()
    assert body["success"] is True
    assert {"engine", "networkMode", "pageCount", "totalMs", "avgConfidence", "pages", "images", "summary"} <= set(body["data"])
    assert body["data"]["pageCount"] == 2


# ---- 검증 오류 ---------------------------------------------------------------------

def test_missing_file_is_422_validation_error(stub):
    r = client.post(URL, data={"engine": "AUTO"})
    assert r.status_code == 422
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_unknown_engine_value_is_422(stub):
    r = post({"engine": "FAKE"})
    assert r.status_code == 422
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_non_pdf_upload_is_400_unsupported_type(stub):
    r = post(pdf=b"just some text", filename="a.pdf")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
    assert stub.calls == []                       # 서비스까지 가지 않는다


def test_pdf_over_page_limit_is_400(stub):
    r = post(pdf=blank_pdf(51))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "PAGE_LIMIT_EXCEEDED"
    assert stub.calls == []


def test_file_over_size_limit_is_413(stub):
    r = post(pdf=b"%PDF-" + b"0" * (20 * 1024 * 1024))
    assert r.status_code == 413
    body = r.json()
    assert body["success"] is False and body["error"]["code"] == "FILE_TOO_LARGE"
    assert stub.calls == []


def test_unexpected_server_error_hides_details():
    s = StubExtractService(error=RuntimeError("secret internal detail"))
    app.dependency_overrides[get_extract_service] = lambda: s
    try:
        r = post()
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500
    assert r.json()["code"] == "INTERNAL_SERVER_ERROR"
    assert "secret" not in r.text


# ---- 실제 서비스 연결(OCR 엔진 없이 텍스트 레이어만) ------------------------------------------

def test_pdfium_text_layer_end_to_end():
    """의존성 대역 없이 실제 PDFium 으로 텍스트 레이어를 추출한다. (Tesseract/OpenAI 호출 없음)"""
    r = post({"engine": "PDFIUM"}, pdf=text_pdf())
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["engine"] == "PDFIUM"
    assert data["pageCount"] == 1
    assert data["avgConfidence"] is None
    page = data["pages"][0]
    assert (page["pageNo"], page["source"], page["confidence"]) == (1, "TEXT_LAYER", None)
    assert page["text"] == TEXT_LAYER_SENTENCE


def test_end_to_end_extract_images_returns_png():
    r = post({"engine": "PDFIUM", "extract_images": "true"}, pdf=text_pdf())
    image = r.json()["data"]["images"][0]
    assert (image["pageNo"], image["format"]) == (1, "png")
    assert image["width"] > 0 and image["dataBase64"]


def test_engine_not_allowed_error_shape():
    from app.core.config import get_settings
    from tests.helpers import make_settings
    closed = make_settings(network_env="CLOSED", engine="TESSERACT")
    app.dependency_overrides[get_settings] = lambda: closed
    try:
        r = post({"engine": "OPENAI"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "ENGINE_NOT_ALLOWED"
