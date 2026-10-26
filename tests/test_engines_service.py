import pytest

from app.services.engines_service import EnginesService
from tests.helpers import make_settings


def names(data):
    return [e.name for e in data.engines]


def by_name(data):
    return {e.name: e for e in data.engines}


def test_lists_all_declared_engines_in_order():
    data = EnginesService(make_settings()).get_engines()
    assert names(data) == ["AUTO", "PDFIUM", "TESSERACT", "PADDLE", "OPENAI"]


def test_open_network_everything_available():
    data = EnginesService(make_settings()).get_engines()
    assert data.network_mode == "OPEN"
    assert data.default_engine == "OPENAI"
    assert all(e.available and e.reason is None for e in data.engines)


def test_only_openai_is_external():
    data = EnginesService(make_settings()).get_engines()
    assert {n for n, e in by_name(data).items() if e.external} == {"OPENAI"}


def test_closed_network_disables_openai_with_reason():
    s = make_settings(network_env="CLOSED", engine="TESSERACT")
    data = EnginesService(s).get_engines()
    openai = by_name(data)["OPENAI"]
    assert data.network_mode == "CLOSED"
    assert openai.available is False
    assert "폐쇄망" in openai.reason
    assert by_name(data)["TESSERACT"].available is True


@pytest.mark.parametrize("field,expected", [
    ("openai_api_key", "OPENAI_API_KEY"),
    ("openai_model", "OPENAI_MODEL"),
])
def test_missing_openai_settings_explained(field, expected):
    s = make_settings(engine="AUTO", **{field: None})
    openai = by_name(EnginesService(s).get_engines())["OPENAI"]
    assert openai.available is False
    assert expected in openai.reason


def test_response_uses_camel_case_keys():
    dumped = EnginesService(make_settings()).get_engines().model_dump(by_alias=True)
    assert set(dumped) == {"networkMode", "defaultEngine", "engines"}
    assert set(dumped["engines"][0]) == {"name", "available", "external", "reason"}
