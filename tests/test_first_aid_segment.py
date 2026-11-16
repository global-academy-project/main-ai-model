"""TBM 응급조치 줄 2차: 항목 안 첫 문장만(FIRST_AID_FIRST_SEGMENT_ONLY), 물질 이름은 입력 표기(FIRST_AID_USE_QUERY_NAME).

문장 모양은 11/16 화면에서 본 실제 출력(톨루엔·메테인)을 따른다. 항목 안 문장은 공단 원문에서 '|'로 이어져 있다.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from app.agent import draft_extras as extras
from app.agent import drafting
from app.schemas.agent_work import MsdsDto

TOLUENE_AID = "\n".join([
    "눈에 들어갔을 때: 긴급 의료조치를 받으시오|물질과 접촉시 즉시 20분 이상 흐르는 물에 피부와 눈을 씻어내시오",
    "흡입했을 때: 노출되거나 노출이 우려되면 의학적인 조치·조언을 구하시오.|토하게 하지 마시오.",
])
METHANE_AID = "\n".join([
    "눈에 들어갔을 때: 눈에 묻으면 몇 분간 물로 조심해서 씻으시오",
    "흡입했을 때: 신선한 공기가 있는 곳으로 옮기시오|긴급 의료조치를 받으시오|호흡하지 않는 경우 인공호흡을 실시하시오|호흡이 힘들 경우 산소를 공급하시오",
])


@pytest.fixture(autouse=True)
def switches(monkeypatch):
    for name in ("ADD_FIRST_AID_POINTS", "PREFER_INHALATION_FIRST_AID", "FIRST_AID_FIRST_SEGMENT_ONLY", "FIRST_AID_USE_QUERY_NAME"):
        monkeypatch.setattr(extras, name, True)


def item(query: str, name: str, ghs: list[str], first_aid: str | None) -> MsdsDto:
    return MsdsDto.model_validate({"query": query, "found": True, "name": name, "ghs_classes": ghs, "first_aid": first_aid})


def one(m: MsdsDto) -> str:
    (point,) = extras.first_aid_points([m])
    return point


def test_methane_real_shape_first_sentence_and_input_name():
    assert one(item("메탄", "메테인", ["인화성 가스", "고압가스"], METHANE_AID)) == "메탄 응급조치: 흡입했을 때: 신선한 공기가 있는 곳으로 옮기시오"


def test_toluene_real_shape_drops_the_stray_swallow_advice():
    point = one(item("톨루엔", "톨루엔", ["인화성 액체"], TOLUENE_AID))
    assert point == "톨루엔 응급조치: 흡입했을 때: 노출되거나 노출이 우려되면 의학적인 조치·조언을 구하시오."
    assert "토하게" not in point and "|" not in point


def test_fallback_first_line_also_uses_first_sentence():
    point = one(item("가성소다", "수산화나트륨", ["피부 부식성/피부 자극성"], TOLUENE_AID))
    assert point == "가성소다 응급조치: 눈에 들어갔을 때: 긴급 의료조치를 받으시오"


def test_label_only_first_part_takes_the_next_sentence():
    assert one(item("톨루엔", "톨루엔", ["인화성 액체"], "흡입했을 때:|신선한 공기로 옮기시오|그 밖")).endswith("흡입했을 때: 신선한 공기로 옮기시오")


def test_empty_pieces_give_no_point():
    assert extras.first_aid_points([item("톨루엔", "톨루엔", ["인화성 액체"], "|"), item("아세톤", "아세톤", ["인화성 액체"], " | ")]) == []


def test_no_pipe_text_is_unchanged():
    assert one(item("톨루엔", "톨루엔", ["인화성 액체"], "흡입했을 때: 공기를 마시게 하시오.")).endswith("흡입했을 때: 공기를 마시게 하시오.")


def test_long_first_sentence_is_still_limited_to_80_chars():
    long = "흡입했을 때: " + "신선한 공기가 있는 곳으로 옮기고 안정을 취하며 따뜻하게 유지하시오 " * 4
    summary = one(item("톨루엔", "톨루엔", ["인화성 액체"], long)).removeprefix("톨루엔 응급조치: ")
    assert len(summary) <= extras.FIRST_AID_MAX_CHARS and summary.endswith("…")


def test_segment_switch_off_keeps_pipes(monkeypatch):
    monkeypatch.setattr(extras, "FIRST_AID_FIRST_SEGMENT_ONLY", False)
    point = one(item("메탄", "메테인", ["인화성 가스"], METHANE_AID))
    assert "|" in point and point.startswith("메탄 응급조치: 흡입했을 때: 신선한 공기가 있는 곳으로 옮기시오|")


def test_name_uses_query_and_falls_back_to_official_name():
    assert one(item("프로판", "프로페인", ["인화성 가스"], METHANE_AID)).startswith("프로판 응급조치:")
    assert one(item("", "프로페인", ["인화성 가스"], METHANE_AID)).startswith("프로페인 응급조치:")


def test_name_switch_off_uses_official_name(monkeypatch):
    monkeypatch.setattr(extras, "FIRST_AID_USE_QUERY_NAME", False)
    assert one(item("메탄", "메테인", ["인화성 가스"], METHANE_AID)).startswith("메테인 응급조치:")
    assert one(item("메탄", "", ["인화성 가스"], METHANE_AID)).startswith("메탄 응급조치:")


def test_both_switches_off_is_the_first_patch_behavior(monkeypatch):
    monkeypatch.setattr(extras, "FIRST_AID_FIRST_SEGMENT_ONLY", False)
    monkeypatch.setattr(extras, "FIRST_AID_USE_QUERY_NAME", False)
    point = one(item("메탄", "메테인", ["인화성 가스"], METHANE_AID))
    assert point == f"메테인 응급조치: {extras._shorten(METHANE_AID.splitlines()[1])}"


def test_mock_results_are_unchanged_by_both_switches(monkeypatch):
    from app.tools.mock.mock_msds import lookup_msds

    items = [MsdsDto.model_validate(r) for r in lookup_msds({"materials": ["톨루엔", "메탄올", "아세톤", "수소"]})["results"]]
    on = extras.first_aid_points(items)
    monkeypatch.setattr(extras, "FIRST_AID_FIRST_SEGMENT_ONLY", False)
    monkeypatch.setattr(extras, "FIRST_AID_USE_QUERY_NAME", False)
    assert extras.first_aid_points(items) == on


def test_no_pipe_in_any_point_when_on():
    points = extras.first_aid_points([item("톨루엔", "톨루엔", ["인화성 액체"], TOLUENE_AID), item("메탄", "메테인", ["인화성 가스"], METHANE_AID)])
    assert len(points) == 2 and not any("|" in p for p in points)


def test_tbm_data_shows_input_name_and_first_sentence():
    tbm = drafting.tbm_data([], [], msds=[item("메탄", "메테인", ["인화성 가스"], METHANE_AID)])
    assert "메탄 응급조치: 흡입했을 때: 신선한 공기가 있는 곳으로 옮기시오" in [p["point"] for p in tbm["points"]]


def test_through_wrapper_keeps_query_for_input_name():
    """래퍼가 query를 요청 이름으로 되돌리므로 TBM 줄에도 입력 표기가 나온다 (메탄 → 74-82-8 조회, 공단 이름 메테인)."""
    from app.agent.msds_setup import make_msds_lookup

    class Client:
        def lookup(self, arguments):
            (term,) = arguments["materials"]
            return {"results": [{"query": term, "found": True, "name": "메테인", "cas_no": "74-82-8", "ghs_classes": ["인화성 가스 : 구분1", "자료없음"],
                                 "first_aid": METHANE_AID, "hazards": [], "regulations": [], "regulatory_tags": [], "source": "공단"}]}

    raw = make_msds_lookup(Client())({"materials": ["메탄"]})["results"][0]
    assert raw["name"] == "메테인"
    assert one(MsdsDto.model_validate(raw)).startswith("메탄 응급조치: 흡입했을 때:")


def test_switches_are_on_by_default():
    path = pathlib.Path(extras.__file__)
    spec = importlib.util.spec_from_file_location("draft_extras_fresh2", path)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.FIRST_AID_FIRST_SEGMENT_ONLY is True and fresh.FIRST_AID_USE_QUERY_NAME is True
