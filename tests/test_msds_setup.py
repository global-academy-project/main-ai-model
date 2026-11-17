"""lookup_msds 실제 도구 등록 시험 (app/agent/msds_setup.py + 설정·등록·허가서 매칭 패치).

공단 API는 부르지 않는다. 가짜 클라이언트, 또는 실제 KoshaMsdsClient의 _request_items만 가짜로 바꾼다.
ghs_classes 문자열은 11/16 PC 실측(probe_msds)에서 받은 값 그대로다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agent import msds_setup
from app.agent.msds_setup import BUDGET_CODE, LOOKUP_TERMS, build_msds_tool, ghs_labels, make_msds_lookup
from app.agent.text_rules import MATERIALS
from app.core import Settings
from app.exceptions import ToolError
from app.tools.mock.mock_msds import MSDS_PARAMETERS as MOCK_PARAMETERS
from app.tools.mock.mock_msds import lookup_msds as mock_lookup
from app.tools.tool_setup import build_tool_registry

ROOT = Path(__file__).resolve().parent.parent

# ── 실측 문자열 (11/16) ──
TOLUENE = "인화성 액체 : 구분2|피부 부식성/피부 자극성 : 구분2|생식독성 : 구분2|특정표적장기 독성(1회 노출) : 구분3(마취영향)|특정표적장기 독성(반복 노출) : 구분2|흡인 유해성 : 구분1"
BENZENE_CUT = "인화성 액체 : 구분2|피부 부식성/피부 자극성 : 구분2|심한 눈 손상성/눈 자극성 : 구분2|생식세포 변이원성 : 구분1A|발암성 : 구분1B|특정표적장기 독성(반복 노출) : 구분1|흡인 유해성 : 구분1|만성"
CHLORINE_CUT = "산화성 가스 : 구분1|고압가스 : 액화가스|급성 독성(경구) : 구분4|급성 독성(흡입: 가스) : 구분2|피부 부식성/피부 자극성 : 구분1|심한 눈 손상성/눈 자극성 : 구분1|특정표적장기 독성(1회 노출) :"
XYLENE_CUT = "인화성 액체 : 구분3|급성 독성(흡입: 증기) : 구분4|피부 부식성/피부 자극성 : 구분2|심한 눈 손상성/눈 자극성 : 구분2|생식독성 : 구분2|특정표적장기 독성(1회 노출) : 구분3(마취영향)|특정표적장기"
HYDROGEN = "인화성 가스 : 구분1|고압가스 : 압축가스"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("MSDS_TOOL", "KOSHA_MSDS_SERVICE_KEY", "CHECK_RULES_TOOL", "NETWORK_ENV", "RAG_ENABLED"):
        monkeypatch.delenv(name, raising=False)


def settings(**kwargs) -> Settings:
    """msds_tool을 안 주면 MOCK. 코드 기본값(KOSHA)은 test_settings_default_is_kosha_and_needs_key가 따로 본다."""
    return Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK", **{"msds_tool": "MOCK", **kwargs})


class FakeClient:
    """client.lookup({"materials": [조회어]})를 흉내 낸다. data: 조회어 → 결과 dict | ToolError."""

    def __init__(self, data: dict[str, object], on_call=None) -> None:
        self.data = data
        self.calls: list[str] = []
        self.on_call = on_call

    def lookup(self, arguments):
        (term,) = arguments["materials"]
        self.calls.append(term)
        if self.on_call:
            self.on_call()
        value = self.data.get(term, {"query": term, "found": False})
        if isinstance(value, ToolError):
            raise value
        return {"results": [dict(value, query=term)]}


def found(name: str, cas: str, ghs=("자료없음",), **extra) -> dict:
    return {"found": True, "name": name, "cas_no": cas, "ghs_classes": list(ghs), "hazards": ["h"], "first_aid": "f",
            "regulations": [], "regulatory_tags": [], "source": "한국산업안전보건공단 물질안전보건자료 조회 서비스", **extra}


def upstream() -> ToolError:
    return ToolError("MSDS_UPSTREAM_ERROR", "공단 MSDS API에 연결하거나 응답을 처리하지 못했습니다.", tool="lookup_msds")


# ── ghs_classes 정리 ──


def test_ghs_labels_toluene_with_no_data_entry():
    assert ghs_labels([TOLUENE, "자료없음"]) == [
        "인화성 액체", "피부 부식성/피부 자극성", "생식독성", "특정표적장기 독성(1회 노출)", "특정표적장기 독성(반복 노출)", "흡인 유해성",
    ]


def test_ghs_labels_drop_truncated_tail():
    assert ghs_labels([BENZENE_CUT])[-1] == "흡인 유해성"  # 끝의 "만성" 조각
    assert "만성" not in ghs_labels([BENZENE_CUT])
    assert ghs_labels([CHLORINE_CUT])[-1] == "심한 눈 손상성/눈 자극성"
    assert "특정표적장기 독성(1회 노출)" not in ghs_labels([CHLORINE_CUT])  # 값 없이 잘린 항목
    assert ghs_labels([XYLENE_CUT])[-1] == "특정표적장기 독성(1회 노출)"  # 마지막 "특정표적장기" 조각은 버림


def test_ghs_labels_keep_colon_inside_label():
    labels = ghs_labels([CHLORINE_CUT])
    assert "급성 독성(흡입: 가스)" in labels  # 라벨 안의 콜론은 구분이 아니다
    assert labels[0] == "산화성 가스"
    assert not any(label.endswith("(흡입") for label in labels)


def test_ghs_labels_gases():
    assert ghs_labels([HYDROGEN, "자료없음"]) == ["인화성 가스", "고압가스"]


def test_ghs_labels_ignore_not_applicable_and_bad_input():
    assert ghs_labels(["인화성 액체 : 해당없음|생식독성 : 구분2"]) == ["생식독성"]
    assert ghs_labels(["인화성 액체 : 분류되지 않음"]) == []
    assert ghs_labels(["자료없음"]) == []
    assert ghs_labels([]) == []
    assert ghs_labels(None) == []
    assert ghs_labels("인화성 액체 : 구분2") == []  # 목록이 아니면 읽지 않는다
    assert ghs_labels([None, 3, " : 구분2"]) == []
    assert ghs_labels(["인화성 액체 : 구분2|인화성 액체 : 구분3"]) == ["인화성 액체"]  # 같은 라벨은 한 번


def test_labels_have_no_pipe_or_division():
    for entry in (TOLUENE, BENZENE_CUT, CHLORINE_CUT, XYLENE_CUT, HYDROGEN):
        for label in ghs_labels([entry]):
            assert "|" not in label and "구분" not in label


# ── 조회어 표 ──


def test_lookup_terms_only_for_dictionary_names():
    assert set(LOOKUP_TERMS) <= set(MATERIALS)
    assert len(LOOKUP_TERMS) == 18
    assert LOOKUP_TERMS["메탄올"] == "메틸알코올"
    assert LOOKUP_TERMS["자일렌"] == "크실렌"
    assert LOOKUP_TERMS["휘발유"] == "가솔린"
    assert "시너" not in LOOKUP_TERMS  # 혼합물이라 못 찾음이 맞다


def test_cas_terms_from_second_probe():
    assert {k: LOOKUP_TERMS[k] for k in ("프로판", "메탄", "질소", "경유")} == {
        "프로판": "74-98-6", "메탄": "74-82-8", "질소": "7727-37-9", "경유": "68334-30-5",
    }
    cas = re.compile(r"^\d{2,7}-\d{2}-\d$")
    for name, term in LOOKUP_TERMS.items():
        if cas.match(term):  # CAS 형식이면 검산 자리(체크섬)까지 맞는다
            digits = term.replace("-", "")
            check = sum(int(d) * w for d, w in zip(reversed(digits[:-1]), range(1, len(digits)))) % 10
            assert check == int(digits[-1]), f"{name} {term}"


def test_cas_term_is_sent_to_client_and_name_stays_official():
    client = FakeClient({"74-98-6": found("프로페인", "74-98-6", ghs=["인화성 가스 : 구분1|고압가스 : 액화가스", "자료없음"])})
    result = make_msds_lookup(client)({"materials": ["프로판"]})["results"][0]
    assert client.calls == ["74-98-6"]
    assert result["query"] == "프로판" and result["name"] == "프로페인"
    assert result["ghs_classes"] == ["인화성 가스", "고압가스"]


# ── 래퍼 동작 ──


def test_term_mapping_and_query_restored():
    client = FakeClient({"메틸알코올": found("메틸 알코올", "67-56-1", ghs=[TOLUENE, "자료없음"])})
    result = make_msds_lookup(client)({"materials": ["메탄올"]})["results"][0]
    assert client.calls == ["메틸알코올"]  # 공단에서 찾히는 조회어로 부른다
    assert result["query"] == "메탄올"  # 요청받은 이름으로 되돌린다
    assert result["name"] == "메틸 알코올"  # 공단 이름은 그대로
    assert result["found"] is True and result["cas_no"] == "67-56-1"
    assert result["ghs_classes"][0] == "인화성 액체"
    assert result["ghs_detail"] == [TOLUENE, "자료없음"]  # 원래 값은 남긴다


def test_unknown_name_passes_through_and_not_found_is_plain():
    client = FakeClient({})
    out = make_msds_lookup(client)({"materials": ["시너", "사전밖물질"]})
    assert client.calls == ["시너", "사전밖물질"]
    assert out["results"] == [{"query": "시너", "found": False}, {"query": "사전밖물질", "found": False}]


def test_output_keeps_mock_keys_and_registry_validates_into_dto():
    from app.schemas.agent_work import MsdsDto

    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3", ghs=[TOLUENE, "자료없음"])})
    result = make_msds_lookup(client)({"materials": ["톨루엔"]})["results"][0]
    mock_keys = set(mock_lookup({"materials": ["톨루엔"]})["results"][0])
    assert mock_keys <= set(result)  # mock이 주던 키는 모두 있다
    dto = MsdsDto.model_validate(result)  # ghs_detail 같은 덧붙인 키는 DTO가 무시한다
    assert dto.query == "톨루엔" and dto.found and dto.ghs_classes[0] == "인화성 액체"
    assert not hasattr(dto, "ghs_detail")


def test_partial_failure_keeps_found_materials():
    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3"), "아세톤": upstream(), "벤젠": found("벤젠", "71-43-2")})
    results = make_msds_lookup(client)({"materials": ["톨루엔", "아세톤", "벤젠"]})["results"]
    assert [r["found"] for r in results] == [True, False, True]
    assert results[1] == {"query": "아세톤", "found": False, "lookup_error": "MSDS_UPSTREAM_ERROR"}
    assert [r["query"] for r in results] == ["톨루엔", "아세톤", "벤젠"]


def test_all_attempts_failed_raises_original_error():
    client = FakeClient({"톨루엔": upstream(), "벤젠": upstream()})
    with pytest.raises(ToolError) as info:
        make_msds_lookup(client)({"materials": ["톨루엔", "벤젠"]})
    assert info.value.code == "MSDS_UPSTREAM_ERROR"
    assert info.value.tool == "lookup_msds"


def test_not_found_is_not_a_failure():
    client = FakeClient({})
    assert make_msds_lookup(client)({"materials": ["시너"]})["results"] == [{"query": "시너", "found": False}]


def test_budget_exceeded_skips_remaining_materials():
    now = [0.0]
    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3"), "벤젠": found("벤젠", "71-43-2")}, on_call=lambda: now.__setitem__(0, now[0] + 20))
    lookup = make_msds_lookup(client, budget_seconds=25.0, clock=lambda: now[0])
    results = lookup({"materials": ["톨루엔", "벤젠", "아세톤", "헥산"]})["results"]
    assert client.calls == ["톨루엔", "벤젠"]  # 20초, 40초 → 두 번째 조회 뒤에 상한을 넘었다
    assert [r["found"] for r in results] == [True, True, False, False]
    assert results[2] == {"query": "아세톤", "found": False, "lookup_error": BUDGET_CODE}


def test_budget_exceeded_after_only_failures_still_raises():
    now = [0.0]
    client = FakeClient({"톨루엔": upstream()}, on_call=lambda: now.__setitem__(0, now[0] + 30))
    lookup = make_msds_lookup(client, budget_seconds=25.0, clock=lambda: now[0])
    with pytest.raises(ToolError) as info:
        lookup({"materials": ["톨루엔", "벤젠"]})
    assert info.value.code == "MSDS_UPSTREAM_ERROR"
    assert client.calls == ["톨루엔"]


def test_duplicates_are_looked_up_once():
    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3")})
    results = make_msds_lookup(client)({"materials": ["톨루엔", " 톨루엔 ", "TOLUENE-없음", "톨루엔"]})["results"]
    assert client.calls == ["톨루엔", "TOLUENE-없음"]
    assert [r["query"] for r in results] == ["톨루엔", " 톨루엔 ", "TOLUENE-없음", "톨루엔"]
    assert results[1]["found"] is True and results[0] is not results[3]


def test_blank_names_are_not_looked_up():
    client = FakeClient({})
    assert make_msds_lookup(client)({"materials": ["", "  "]})["results"] == [{"query": "", "found": False}, {"query": "  ", "found": False}]
    assert client.calls == []


@pytest.mark.parametrize("bad", [None, "톨루엔", [1], ["톨루엔", None]])
def test_input_validation(bad):
    with pytest.raises(ToolError) as info:
        make_msds_lookup(FakeClient({}))({"materials": bad})
    assert info.value.code == "TOOL_INPUT_INVALID"


def test_too_many_materials():
    with pytest.raises(ToolError) as info:
        make_msds_lookup(FakeClient({}))({"materials": ["물질"] * 21})
    assert info.value.code == "TOOL_INPUT_INVALID"
    assert make_msds_lookup(FakeClient({}))({"materials": ["물질"] * 20})["results"][0]["found"] is False


# ── 실제 KoshaMsdsClient를 거친 전체 경로 (HTTP만 가짜) ──


def _fake_request_items(operation, parameters):
    if operation == "getChemList001":
        word = parameters["searchWrd"]
        table = {"메틸알코올": ("메틸 알코올", "67-56-1", "C1"), "이소프로필알코올": ("이소프로필 알코올", "67-63-0", "C2")}
        if word in table:
            name, cas, chem_id = table[word]
            return [{"chemId": chem_id, "chemNameKor": name, "casNo": cas, "lastDate": "20240101"}]
        return []
    if operation == "getChemDetail021":
        return [{"msdsItemNameKor": "유해성·위험성 분류", "itemDetail": TOLUENE}, {"msdsItemNameKor": "유해성·위험성 분류 2", "itemDetail": "자료없음"}]
    if operation == "getChemDetail041":
        return [{"msdsItemNameKor": "흡입했을 때", "itemDetail": "신선한 공기로 옮긴다."}]
    return [{"msdsItemNameKor": "산업안전보건법에 의한 규제", "itemDetail": "관리대상유해물질"}]


def test_through_real_client_with_fake_http(monkeypatch):
    from app.publicdata.msds import KoshaMsdsClient

    monkeypatch.setattr(KoshaMsdsClient, "_request_items", lambda self, operation, parameters: _fake_request_items(operation, parameters))
    handler, parameters = build_msds_tool(settings(msds_tool="KOSHA", kosha_msds_service_key="TEST-KEY"))
    results = handler({"materials": ["메탄올", "이소프로필알코올", "시너"]})["results"]

    assert [r["query"] for r in results] == ["메탄올", "이소프로필알코올", "시너"]
    assert results[0]["name"] == "메틸 알코올" and results[0]["cas_no"] == "67-56-1"
    assert results[1]["name"] == "이소프로필 알코올"  # 공단 이름은 사용자 표기와 글자가 다르다
    assert results[1]["ghs_classes"][0] == "인화성 액체"
    assert results[0]["regulatory_tags"][0]["code"] == "CONTROLLED_HAZARDOUS_SUBSTANCE"  # A의 태그 분류는 그대로
    assert results[2] == {"query": "시너", "found": False}
    assert parameters["required"] == ["materials"]


# ── 설정 ──


def test_settings_default_is_kosha_and_needs_key():
    # 코드 기본값은 KOSHA다 (config.py). 키가 없으면 기동 때 막는다 (조용히 mock으로 남지 않게).
    assert Settings.model_fields["msds_tool"].default == "KOSHA"
    with pytest.raises(ValidationError, match="KOSHA_MSDS_SERVICE_KEY"):
        Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK")
    s = Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK", kosha_msds_service_key="TEST-KEY")
    assert s.msds_tool == "KOSHA"


def test_settings_mock_needs_no_key():
    s = settings(msds_tool="MOCK")
    assert s.msds_tool == "MOCK" and s.kosha_msds_service_key is None


def test_settings_kosha_requires_key():
    with pytest.raises(ValidationError, match="KOSHA_MSDS_SERVICE_KEY"):
        settings(msds_tool="KOSHA")
    with pytest.raises(ValidationError, match="KOSHA_MSDS_SERVICE_KEY"):
        settings(msds_tool="KOSHA", kosha_msds_service_key="   ")
    s = settings(msds_tool="KOSHA", kosha_msds_service_key="TEST-KEY")
    assert "TEST-KEY" not in repr(s) and "TEST-KEY" not in str(s)  # 키는 SecretStr


def test_settings_read_env_names(monkeypatch):
    monkeypatch.setenv("MSDS_TOOL", "KOSHA")
    monkeypatch.setenv("KOSHA_MSDS_SERVICE_KEY", "TEST-KEY")
    s = Settings(_env_file=None, network_env="OPEN", check_rules_tool="MOCK")
    assert s.msds_tool == "KOSHA" and s.kosha_msds_service_key.get_secret_value() == "TEST-KEY"


def test_settings_unknown_msds_tool_is_rejected():
    with pytest.raises(ValidationError):
        settings(msds_tool="REAL")


def test_build_msds_tool_mock_returns_none():
    assert build_msds_tool(settings()) is None


def test_build_msds_tool_kosha_without_key_is_refused_even_if_validator_bypassed():
    class Bare:
        msds_tool = "KOSHA"
        kosha_msds_service_key = None

    with pytest.raises(ValueError, match="KOSHA_MSDS_SERVICE_KEY"):
        build_msds_tool(Bare())  # type: ignore[arg-type]


def test_mock_does_not_import_a_module():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from app.agent.msds_setup import build_msds_tool\n"
        "class S:\n"
        "    msds_tool = 'MOCK'\n"
        "    kosha_msds_service_key = None\n"
        "assert build_msds_tool(S()) is None\n"
        "assert 'app.publicdata.msds' not in sys.modules, 'A 모듈을 불러왔습니다'\n"
    )
    done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr


# ── 도구 등록 ──


def _by_name(registry):
    return {t.name: t for t in registry.tools()}


def test_registry_default_is_unchanged_mock():
    tools = _by_name(build_tool_registry())
    assert tools["lookup_msds"].mock is True and tools["lookup_msds"].handler is mock_lookup
    assert tools["lookup_msds"].parameters == MOCK_PARAMETERS
    assert set(tools) == {"classify_work", "fill_form", "search_accidents", "search_guides", "lookup_msds", "check_rules"}


def test_registry_with_msds_tool_replaces_only_lookup_msds():
    def handler(arguments):
        return {"results": []}

    params = {"type": "object", "properties": {}}
    base = _by_name(build_tool_registry())
    tools = _by_name(build_tool_registry(msds_tool=(handler, params)))
    assert tools["lookup_msds"].handler is handler
    assert tools["lookup_msds"].mock is False and tools["lookup_msds"].owner == "A"
    assert tools["lookup_msds"].parameters == params
    for name in set(base) - {"lookup_msds"}:
        assert tools[name].handler is base[name].handler and tools[name].mock == base[name].mock
    assert set(tools) == set(base)


def test_agent_deps_follows_settings(monkeypatch):
    import app.agent.agent_deps as deps

    # RAG는 끈다 (이 시험은 lookup_msds만 본다. 켜 두면 EMBEDDING_URL 이 없는 환경에서 검색 백엔드를 만들다 실패한다)
    off = {"rag_enabled": False}
    for value, real in ((settings(**off), False), (settings(msds_tool="KOSHA", kosha_msds_service_key="TEST-KEY", **off), True)):
        monkeypatch.setattr(deps, "get_settings", lambda value=value: value)
        deps.get_tool_registry.cache_clear()
        try:
            tool = _by_name(deps.get_tool_registry())["lookup_msds"]
        finally:
            deps.get_tool_registry.cache_clear()
        assert tool.mock is (not real)
        assert (tool.handler is not mock_lookup) is real


def test_agent_deps_no_longer_imports_publicdata():
    source = (ROOT / "app" / "agent" / "agent_deps.py").read_text(encoding="utf-8")
    assert "register_publicdata_msds" not in source
    assert "build_msds_tool" in source


# ── 허가서 물질 매칭 (R4) ──


def _permit_category(msds_items, nearby):
    from app.agent import drafting
    from app.schemas.agent_work import MsdsDto, WorkSlots

    slots = WorkSlots(nearby_materials=nearby, location="3호기", headcount=2)
    dtos = [MsdsDto.model_validate(m) for m in msds_items]
    data = drafting.permit_data("HOT_WORK_PERMIT", slots, [], None, [], dtos)
    return data["nearby_hazardous_materials"]


def test_permit_matches_by_user_name_when_official_name_differs():
    client = FakeClient({"이소프로필알코올": found("이소프로필 알코올", "67-63-0", ghs=["인화성 액체 : 구분2|흡인 유해성 : 구분2", "자료없음"])})
    result = make_msds_lookup(client)({"materials": ["이소프로필알코올"]})["results"]
    (row,) = _permit_category(result, ["이소프로필알코올"])
    assert row["name"] == "이소프로필알코올"
    assert row["cas_no"] == "67-63-0"
    assert row["category"] == "인화성 액체"  # 덩어리가 아니라 라벨 하나


def test_permit_case_that_the_old_key_could_not_match():
    """패치 전 키(m.name or m.query)는 공단 이름이 사용자 표기와 다르면 못 찾는다. 그 상황임을 못 박고 새 키로는 찾는지 본다."""
    result = make_msds_lookup(FakeClient({"이소프로필알코올": found("이소프로필 알코올", "67-63-0", ghs=["인화성 액체 : 구분2"])}))({"materials": ["이소프로필알코올"]})["results"]
    assert "이소프로필알코올" not in {r["name"] or r["query"] for r in result}
    (row,) = _permit_category(result, ["이소프로필알코올"])
    assert row["cas_no"] == "67-63-0" and row["category"] == "인화성 액체"


def test_permit_mock_result_is_unchanged():
    mock_items = mock_lookup({"materials": ["톨루엔", "메탄올"]})["results"]
    rows = _permit_category(mock_items, ["톨루엔", "메탄올", "시너"])
    assert [(r["name"], r["cas_no"], r["category"]) for r in rows] == [
        ("톨루엔", "108-88-3", "인화성 액체"), ("메탄올", "67-56-1", "인화성 액체"), ("시너", None, None),
    ]


def test_msds_setup_module_has_no_network_or_key_logging():
    source = (ROOT / "app" / "agent" / "msds_setup.py").read_text(encoding="utf-8")
    assert "urlopen" not in source and "print(" not in source
    assert "get_secret_value" in source and "logger" in source
    assert msds_setup.CALL_TIMEOUT_SECONDS <= 10 and msds_setup.BUDGET_SECONDS < 120


# ── 변형 검사에서 드러난 구멍 보강 ──


def test_client_gets_short_call_timeout(monkeypatch):
    from app.publicdata.msds import KoshaMsdsClient

    seen = {}
    original = KoshaMsdsClient.__init__

    def spy(self, service_key, **kwargs):
        seen.update(kwargs)
        original(self, service_key, **kwargs)

    monkeypatch.setattr(KoshaMsdsClient, "__init__", spy)
    build_msds_tool(settings(msds_tool="KOSHA", kosha_msds_service_key="TEST-KEY"))
    assert seen == {"timeout_seconds": msds_setup.CALL_TIMEOUT_SECONDS}


def test_case_variants_are_looked_up_once():
    client = FakeClient({"Acetone": found("아세톤", "67-64-1")})
    results = make_msds_lookup(client)({"materials": ["Acetone", "ACETONE", "acetone"]})["results"]
    assert client.calls == ["Acetone"]
    assert [r["query"] for r in results] == ["Acetone", "ACETONE", "acetone"]  # 요청한 표기 그대로 돌려준다


def test_warning_log_has_code_only(caplog):
    client = FakeClient({"톨루엔": upstream(), "벤젠": found("벤젠", "71-43-2")})
    with caplog.at_level("WARNING", logger="app.agent.msds_setup"):
        make_msds_lookup(client)({"materials": ["톨루엔", "벤젠"]})
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["MSDS 조회 실패: code=MSDS_UPSTREAM_ERROR"]


def test_all_failed_raises_the_last_error():
    client = FakeClient({"톨루엔": ToolError("MSDS_UPSTREAM_ERROR", "첫째", tool="lookup_msds"), "벤젠": ToolError("MSDS_UNEXPECTED", "둘째", tool="lookup_msds")})
    with pytest.raises(ToolError) as info:
        make_msds_lookup(client)({"materials": ["톨루엔", "벤젠"]})
    assert info.value.code == "MSDS_UNEXPECTED"  # 마지막 실패를 알린다


def test_registry_registers_msds_with_replace(monkeypatch):
    """replace=True가 빠지면 이미 등록된 mock과 겹쳐 등록 오류가 난다. 오류 없이 교체되는지 직접 본다."""
    tools = _by_name(build_tool_registry(msds_tool=(lambda a: {"results": []}, {"type": "object"})))
    assert tools["lookup_msds"].mock is False


# ── 에이전트(WorkRegisterAgent._lookup_msds)까지 이어 본 경로 ──


def _agent_lookup(handler, materials):
    from app.agent.work_register import WorkRegisterAgent

    registry = build_tool_registry(msds_tool=(handler, MOCK_PARAMETERS))
    issues: list = []
    return WorkRegisterAgent._lookup_msds(registry, materials, issues), issues


def test_agent_path_found_and_not_found_issue():
    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3", ghs=[TOLUENE, "자료없음"])})
    msds, issues = _agent_lookup(make_msds_lookup(client), ["톨루엔", "시너"])
    assert [m.query for m in msds] == ["톨루엔", "시너"]
    assert msds[0].found and msds[0].ghs_classes[0] == "인화성 액체" and msds[0].cas_no == "108-88-3"
    assert [i.code for i in issues] == ["MSDS_NOT_FOUND"] and "시너" in issues[0].message


def test_agent_path_all_failed_becomes_msds_unavailable():
    msds, issues = _agent_lookup(make_msds_lookup(FakeClient({"톨루엔": upstream()})), ["톨루엔"])
    assert msds == [] and [i.code for i in issues] == ["MSDS_UNAVAILABLE"]
    assert "MSDS_UPSTREAM_ERROR" in issues[0].message


def test_agent_path_partial_failure_keeps_found_and_reports_missing():
    client = FakeClient({"톨루엔": found("톨루엔", "108-88-3"), "아세톤": upstream()})
    msds, issues = _agent_lookup(make_msds_lookup(client), ["톨루엔", "아세톤"])
    assert [m.found for m in msds] == [True, False]
    assert [i.code for i in issues] == ["MSDS_NOT_FOUND"] and "아세톤" in issues[0].message
