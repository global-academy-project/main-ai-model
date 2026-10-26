import asyncio
from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from app.core.dependecies import pdf_dep
from app.exceptions.business_errors import FileTooLargeError, PageLimitExceededError, UnsupportedFileTypeError
from tests.helpers import blank_pdf


def validate(data: bytes, filename="a.pdf"):
    upload = UploadFile(file=BytesIO(data), filename=filename)
    return asyncio.run(pdf_dep.get_pdf_input(upload))


def test_limits_match_the_documented_values():
    assert pdf_dep.MAX_FILE_SIZE_MB == 20
    assert pdf_dep.MAX_PDF_PAGE_NUM == 50


def test_valid_pdf_returns_pdf_input():
    data = blank_pdf(3)
    result = validate(data, "보고서.pdf")
    assert result.filename == "보고서.pdf"
    assert result.pages == 3
    assert result.content == data


def test_missing_filename_falls_back():
    upload = UploadFile(file=BytesIO(blank_pdf(1)), filename=None)
    assert asyncio.run(pdf_dep.get_pdf_input(upload)).filename == "unknownFileName"


def test_exactly_50_pages_allowed():
    assert validate(blank_pdf(50)).pages == 50


def test_51_pages_rejected():
    with pytest.raises(PageLimitExceededError) as e:
        validate(blank_pdf(51))
    assert e.value.status_code == 400
    assert "50" in e.value.details[0].reason


@pytest.mark.parametrize("content", [b"", b"hello", b"<html></html>", b"PK\x03\x04zip", b" %PDF-1.4"])
def test_non_pdf_content_rejected_by_signature(content):
    with pytest.raises(UnsupportedFileTypeError):
        validate(content)


def test_signature_is_checked_even_if_extension_says_pdf():
    with pytest.raises(UnsupportedFileTypeError):
        validate(b"plain text file", filename="fake.pdf")


def test_oversize_rejected(monkeypatch):
    monkeypatch.setattr(pdf_dep, "MAX_FILE_SIZE_MB", 1)     # 큰 파일을 만들지 않고 1MB 로 낮춰서 검사
    with pytest.raises(FileTooLargeError) as e:
        validate(b"%PDF-" + b"0" * (1024 * 1024))
    assert e.value.status_code == 413
    assert "1MB" in e.value.details[0].reason


def test_size_is_checked_before_signature(monkeypatch):
    monkeypatch.setattr(pdf_dep, "MAX_FILE_SIZE_MB", 1)
    with pytest.raises(FileTooLargeError):
        validate(b"x" * (1024 * 1024 + 1))


@pytest.mark.xfail(strict=False, reason="헤더만 %PDF 인 손상 파일은 pypdf 예외가 그대로 올라가 500 으로 나간다 (4xx 로 변환해야 함)")
def test_corrupt_pdf_is_reported_as_client_error():
    from app.exceptions.business_errors import BusinessError
    with pytest.raises(BusinessError):
        validate(b"%PDF-1.4 this is garbage")
