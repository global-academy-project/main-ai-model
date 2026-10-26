import pytest

from app.core.dependecies.engine_dep import get_engine, get_extract_options
from app.exceptions.business_errors import EngineNotAllowedError
from tests.helpers import make_settings


def test_missing_engine_uses_server_default():
    assert get_engine(make_settings(engine="OPENAI"), None) == "OPENAI"
    assert get_engine(make_settings(engine="AUTO"), None) == "AUTO"


@pytest.mark.parametrize("name", ["AUTO", "PDFIUM", "TESSERACT", "PADDLE", "OPENAI"])
def test_requested_engine_is_returned_when_allowed(name):
    assert get_engine(make_settings(), name) == name


def test_external_engine_blocked_in_closed_network():
    s = make_settings(network_env="CLOSED", engine="TESSERACT")
    with pytest.raises(EngineNotAllowedError) as e:
        get_engine(s, "OPENAI")
    assert e.value.status_code == 400
    assert e.value.code == "ENGINE_NOT_ALLOWED"
    assert "OPENAI" in e.value.details[0].reason


def test_openai_blocked_when_key_not_configured():
    s = make_settings(engine="AUTO", openai_api_key=None)
    with pytest.raises(EngineNotAllowedError):
        get_engine(s, "OPENAI")


def test_local_engines_still_allowed_in_closed_network():
    s = make_settings(network_env="CLOSED", engine="TESSERACT")
    assert get_engine(s, "PADDLE") == "PADDLE"


def test_extract_options_defaults():
    o = get_extract_options(engine="AUTO")
    assert (o.engine, o.extract_images, o.lang) == ("AUTO", False, ["ko", "en"])


def test_extract_options_normalizes_lang_and_keeps_flags():
    o = get_extract_options(engine="TESSERACT", extract_images=True, lang=["ko,en"])
    assert (o.engine, o.extract_images, o.lang) == ("TESSERACT", True, ["ko", "en"])
