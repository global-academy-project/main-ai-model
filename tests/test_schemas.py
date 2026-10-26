import pytest
from pydantic import ValidationError

from app.schemas.error import ErrorBody, ErrorDetail, ErrorResponse
from app.schemas.extract import ExtractData, ExtractOptions, ImageInfo, PageInfo, PageSource


class TestExtractOptions:
    def test_defaults(self):
        o = ExtractOptions()
        assert (o.engine, o.extract_images, o.lang) == ("AUTO", False, ["ko", "en"])

    @pytest.mark.parametrize("raw,expected", [
        ("ko,en", ["ko", "en"]),
        (["ko,en"], ["ko", "en"]),
        (["ko", "en"], ["ko", "en"]),
        ([" KO ", "en", "ko"], ["ko", "en"]),     # 공백 제거, 소문자, 중복 제거
        (["en"], ["en"]),
        ([], ["ko", "en"]),                       # 비어 있으면 기본값
        ("", ["ko", "en"]),
        (None, ["ko", "en"]),
        (["", " , "], ["ko", "en"]),
    ])
    def test_lang_normalized(self, raw, expected):
        assert ExtractOptions(lang=raw).lang == expected

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValidationError):
            ExtractOptions(engine="FAKE")

    def test_default_lang_list_is_not_shared_between_instances(self):
        a, b = ExtractOptions(), ExtractOptions()
        a.lang.append("fr")
        assert b.lang == ["ko", "en"]


class TestPageInfo:
    def make(self, **kw):
        base = dict(page_no=1, source=PageSource.OCR, text="안녕", confidence=0.5, processing_ms=12)
        base.update(kw)
        return PageInfo(**base)

    def test_serializes_with_camel_case_and_enum_value(self):
        d = self.make().model_dump(by_alias=True, mode="json")
        assert d == {"pageNo": 1, "source": "OCR", "text": "안녕", "confidence": 0.5, "processingMs": 12}

    def test_none_confidence_stays_none(self):
        assert self.make(confidence=None).model_dump()["confidence"] is None

    @pytest.mark.xfail(strict=False, reason="명세는 신뢰도 소수 4자리(DB DECIMAL(5,4))인데 직렬화가 round(value, 1)이라 0.94 가 0.9 로 나간다")
    def test_confidence_keeps_four_decimals(self):
        assert self.make(confidence=0.9412).model_dump()["confidence"] == pytest.approx(0.9412)


class TestImageInfo:
    def test_format_is_lowercased(self):
        img = ImageInfo(page_no=1, format="PNG", width=1, height=1, data_base64="AA==")
        assert img.format == "png"

    def test_serializes_with_camel_case(self):
        d = ImageInfo(page_no=2, format="png", width=3, height=4, data_base64="AA==").model_dump(by_alias=True)
        assert d == {"pageNo": 2, "format": "png", "width": 3, "height": 4, "dataBase64": "AA=="}


class TestExtractData:
    def make(self, **kw):
        base = dict(engine="TESSERACT", network_mode="OPEN", page_count=1, total_ms=10.0, avg_confidence=None)
        base.update(kw)
        return ExtractData(**base)

    def test_camel_case_keys(self):
        d = self.make().model_dump(by_alias=True)
        assert {"engine", "networkMode", "pageCount", "totalMs", "avgConfidence", "pages", "images", "summary"} <= set(d)

    def test_lists_default_to_empty_and_are_not_shared(self):
        a, b = self.make(), self.make()
        a.pages.append("x")
        assert b.pages == []

    def test_can_be_built_with_snake_case_names(self):
        assert self.make(page_count=7).page_count == 7

    @pytest.mark.xfail(strict=False, reason="요약이 연결되기 전까지 summary 기본값이 임시 문구('임시 텍스트 ')라 Spring 이 그대로 저장한다")
    def test_summary_default_is_not_a_placeholder(self):
        assert "임시" not in self.make().summary


class TestErrorSchemas:
    def test_details_default_is_empty_list_and_independent(self):
        a, b = ErrorBody(code="X", message="m"), ErrorBody(code="Y", message="m")
        a.details.append(ErrorDetail(reason="r"))
        assert b.details == []

    def test_error_response_shape(self):
        d = ErrorResponse(error=ErrorBody(code="X", message="m")).model_dump(mode="json")
        assert d == {"success": False, "error": {"code": "X", "message": "m", "details": []}}
