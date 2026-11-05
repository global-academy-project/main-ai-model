from io import BytesIO
from urllib.parse import parse_qs, urlsplit

from app.publicdata.msds import KoshaMsdsClient, register_publicdata_msds
from app.tools.tool_registry import ToolRegistry


def test_lookup_uses_list_and_analysis_sections(monkeypatch):
    responses = [
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <chemId>1008</chemId><chemNameKor>톨루엔</chemNameKor>
        <casNo>108-88-3</casNo><lastDate>2026-09-17</lastDate>
        </item></items></body></response>""".encode(),
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <msdsItemNameKor>가.유해성·위험성 분류</msdsItemNameKor>
        <itemDetail>인화성 액체</itemDetail>
        </item><item><msdsItemNameKor>나.경고표지</msdsItemNameKor>
        <itemDetail>고인화성 증기</itemDetail>
        </item></items></body></response>""".encode(),
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <msdsItemNameKor>눈에 들어갔을 때</msdsItemNameKor>
        <itemDetail>물로 씻으세요</itemDetail>
        </item></items></body></response>""".encode(),
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <msdsItemNameKor>국내 규제</msdsItemNameKor>
        <itemDetail>관리 대상</itemDetail>
        </item></items></body></response>""".encode(),
    ]
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return BytesIO(responses.pop(0))

    monkeypatch.setattr("app.publicdata.msds.urlopen", fake_urlopen)
    client = KoshaMsdsClient("test service key", timeout_seconds=2)

    result = client.lookup({"materials": ["톨루엔"]})["results"][0]

    assert result["found"] is True
    assert result["cas_no"] == "108-88-3"
    assert result["ghs_classes"] == ["인화성 액체"]
    assert result["hazards"] == [
        "가.유해성·위험성 분류: 인화성 액체",
        "나.경고표지: 고인화성 증기",
    ]
    assert result["first_aid"] == "눈에 들어갔을 때: 물로 씻으세요"
    assert result["regulations"] == ["국내 규제: 관리 대상"]
    assert [urlsplit(request.full_url).path.rsplit("/", 1)[-1] for request in requests] == [
        "getChemList001",
        "getChemDetail021",
        "getChemDetail041",
        "getChemDetail151",
    ]
    assert parse_qs(urlsplit(requests[0].full_url).query)["searchCnd"] == ["0"]
    assert parse_qs(urlsplit(requests[1].full_url).query)["chemId"] == ["1008"]
    assert parse_qs(urlsplit(requests[0].full_url).query)["serviceKey"] == ["test service key"]


def test_lookup_searches_cas_and_uses_short_cache(monkeypatch):
    responses = [
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <chemId>1008</chemId><chemNameKor>톨루엔</chemNameKor>
        <casNo>108-88-3</casNo></item></items></body></response>""".encode(),
        """<response><header><resultCode>00</resultCode></header><body><items><item>
        <msdsItemNameKor>가.분류</msdsItemNameKor>
        <itemDetail>인화성</itemDetail></item></items></body></response>""".encode(),
        b"<response><header><resultCode>00</resultCode></header><body><items/></body></response>",
        b"<response><header><resultCode>00</resultCode></header><body><items/></body></response>",
    ]
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return BytesIO(responses.pop(0))

    monkeypatch.setattr("app.publicdata.msds.urlopen", fake_urlopen)
    client = KoshaMsdsClient("test-key")

    first = client.lookup({"materials": ["108-88-3"]})["results"][0]
    client.lookup({"materials": ["108-88-3"]})

    assert first["found"] is True
    assert parse_qs(urlsplit(requests[0].full_url).query)["searchCnd"] == ["1"]
    assert len(requests) == 4


def test_register_uses_publicdata_implementation_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", "configured-key")
    registry = ToolRegistry()
    registry.register(
        "lookup_msds", lambda arguments: {}, owner="A", description="test", mock=True,
    )

    register_publicdata_msds(registry)

    tool = next(item for item in registry.tools() if item.name == "lookup_msds")
    assert tool.mock is False
    assert tool.description.startswith("공단 MSDS API")