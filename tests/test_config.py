import pytest
from pydantic import ValidationError

from tests.helpers import make_settings


def test_available_engines_keep_declared_order():
    assert make_settings().available_engines == ["AUTO", "PDFIUM", "TESSERACT", "PADDLE", "OPENAI"]


def test_closed_network_rejects_external_engine_at_startup():
    with pytest.raises(ValidationError):
        make_settings(network_env="CLOSED", engine="OPENAI")


def test_closed_network_allows_local_engines():
    s = make_settings(network_env="CLOSED", engine="TESSERACT", default_ocr_engine="PADDLE")
    assert s.network_env == "CLOSED"


def test_closed_network_rejects_external_default_ocr_engine():
    # DEFAULT_OCR_ENGINE 의 기본값이 OPENAI 이므로, 폐쇄망에서는 반드시 명시해야 기동된다
    with pytest.raises(ValidationError):
        make_settings(network_env="CLOSED", engine="TESSERACT", default_ocr_engine="OPENAI")


@pytest.mark.parametrize("missing", ["openai_api_key", "openai_model"])
def test_openai_engine_requires_key_and_model(missing):
    with pytest.raises(ValidationError):
        make_settings(engine="OPENAI", **{missing: None})


def test_openai_settings_not_required_for_other_engines():
    s = make_settings(engine="TESSERACT", openai_api_key=None, openai_model=None)
    assert s.engine == "TESSERACT"


@pytest.mark.parametrize("value", [0, -1, 300.01, 1000])
def test_ai_timeout_out_of_range_rejected(value):
    with pytest.raises(ValidationError):
        make_settings(ai_timeout_seconds=value)


@pytest.mark.parametrize("value", [0.1, 30, 300])
def test_ai_timeout_boundaries_accepted(value):
    assert make_settings(ai_timeout_seconds=value).ai_timeout_seconds == value


@pytest.mark.parametrize("value", [0, -5, 100_001])
def test_ai_max_output_tokens_out_of_range_rejected(value):
    with pytest.raises(ValidationError):
        make_settings(ai_max_output_tokens=value)


@pytest.mark.parametrize("value", [1, 1000, 100_000])
def test_ai_max_output_tokens_boundaries_accepted(value):
    assert make_settings(ai_max_output_tokens=value).ai_max_output_tokens == value


@pytest.mark.parametrize("field,value", [
    ("network_env", "WIFI"),
    ("engine", "FAKE"),          # Settings 에서는 FAKE 를 고를 수 없다
    ("app_env", "staging"),
    ("default_ocr_engine", "PDFIUM"),
    ("pdf_backend", "PYMUPDF"),
])
def test_invalid_literal_values_rejected(field, value):
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_check_engine_ok_when_everything_configured():
    s = make_settings()
    for name in s.available_engines:
        s.check_engine(name)   # 예외 없음


def test_check_engine_openai_messages():
    s = make_settings(engine="AUTO", openai_api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        s.check_engine("OPENAI")
    s = make_settings(engine="AUTO", openai_model=None)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        s.check_engine("OPENAI")


def test_check_engine_closed_network_message():
    s = make_settings(network_env="CLOSED", engine="TESSERACT")
    with pytest.raises(ValueError, match="폐쇄망"):
        s.check_engine("OPENAI")
    s.check_engine("TESSERACT")
    s.check_engine("AUTO")


def test_secret_key_is_masked_in_repr():
    s = make_settings(openai_api_key="sk-very-secret-value")
    assert "sk-very-secret-value" not in repr(s)
    assert "sk-very-secret-value" not in str(s.model_dump())
