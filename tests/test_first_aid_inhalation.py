"""TBM 응급조치 줄: 인화성·흡입독성 물질은 "흡입했을 때" 줄 우선 (draft_extras._first_aid_text).

눈 접촉 줄 문장은 11/16 PC 실측 출력에서 가져왔다. 나머지 줄은 시험용 문장이다 (공단 항목 이름 형태: "항목: 내용", 항목 안은 '|').
"""

from __future__ import annotations

import pytest

from app.agent import draft_extras as extras
from app.agent import drafting
from app.schemas.agent_work import MsdsDto

EYE = "눈에 들어갔을 때: 긴급 의료조치를 받으시오|물질과 접촉시 즉시 20분 이상 흐르는 물에 피부와 눈을 씻어내시오"
SKIN = "피부에 접촉했을 때: 오염된 의복을 벗고 비누와 물로 씻으시오"
INHALE = "흡입했을 때: 신선한 공기가 있는 곳으로 옮기고 호흡하기 쉬운 자세로 안정을 취하시오.|호흡이 힘들면 산소를 공급하시오"
SWALLOW = "먹었을 때: 토하게 하지 마시오"
NOTE = "기타 의사의 주의사항: 증상에 따라 치료하시오"
FIRST_AID = "\n".join([EYE, SKIN, INHALE, SWALLOW, NOTE])


def msds(name: str, ghs: list[str], first_aid: str | None = FIRST_AID, found: bool = True) -> MsdsDto:
    return MsdsDto.model_validate({"query": name, "found": found, "name": name, "ghs_classes": ghs, "first_aid": first_aid})


@pytest.fixture(autouse=True)
def switches(monkeypatch):
    monkeypatch.setattr(extras, "ADD_FIRST_AID_POINTS", True)
    monkeypatch.setattr(extras, "PREFER_INHALATION_FIRST_AID", True)
    # 2차 패치(첫 문장·입력 표기)가 들어 있어도 이 파일은 흡입 줄 고르기만 본다 (2차는 test_first_aid_segment.py)
    monkeypatch.setattr(extras, "FIRST_AID_FIRST_SEGMENT_ONLY", False, raising=False)
    monkeypatch.setattr(extras, "FIRST_AID_USE_QUERY_NAME", False, raising=False)


def only(m: MsdsDto) -> str:
    (point,) = extras.first_aid_points([m])
    return point


def test_flammable_picks_inhalation_line():
    point = only(msds("톨루엔", ["인화성 액체", "생식독성"]))
    assert point.startswith("톨루엔 응급조치: 흡입했을 때: 신선한 공기가 있는 곳으로")
    assert "눈에 들어갔을 때" not in point


def test_inhalation_toxic_but_not_flammable_picks_inhalation():
    assert "흡입했을 때" in only(msds("염소", ["산화성 가스", "급성 독성(흡입: 가스)"]))


def test_neither_flammable_nor_inhalation_keeps_first_line():
    assert "눈에 들어갔을 때" in only(msds("가성소다", ["피부 부식성/피부 자극성", "심한 눈 손상성/눈 자극성"]))


def test_aspiration_hazard_alone_is_not_inhalation():
    """흡인 유해성(삼킨 뒤 폐로 들어감)은 흡입이 아니다."""
    assert "눈에 들어갔을 때" in only(msds("시험", ["흡인 유해성"]))


def test_no_inhalation_line_falls_back_to_first_line():
    text = "\n".join([EYE, SKIN, SWALLOW])
    assert "눈에 들어갔을 때" in only(msds("톨루엔", ["인화성 액체"], text))


def test_only_item_label_counts_not_the_content():
    text = "\n".join([EYE, "먹었을 때: 흡입하지 않게 주의하시오", SKIN])
    assert "눈에 들어갔을 때" in only(msds("톨루엔", ["인화성 액체"], text))


def test_line_without_colon_is_not_an_item():
    assert only(msds("톨루엔", ["인화성 액체"], "흡입 시 신선한 공기로 옮긴다.")).endswith("흡입 시 신선한 공기로 옮긴다.")


def test_mock_first_aid_is_unchanged_with_switch_on_and_off(monkeypatch):
    from app.tools.mock.mock_msds import lookup_msds

    items = [MsdsDto.model_validate(r) for r in lookup_msds({"materials": ["톨루엔", "메탄올", "아세톤", "수소"]})["results"]]
    on = extras.first_aid_points(items)
    monkeypatch.setattr(extras, "PREFER_INHALATION_FIRST_AID", False)
    assert extras.first_aid_points(items) == on


def test_switch_off_is_exactly_the_old_behavior(monkeypatch):
    items = [msds("톨루엔", ["인화성 액체"]), msds("염소", ["급성 독성(흡입: 가스)"]), msds("가성소다", ["피부 부식성/피부 자극성"])]
    monkeypatch.setattr(extras, "PREFER_INHALATION_FIRST_AID", False)
    assert extras.first_aid_points(items, limit=3) == [f"{m.name} 응급조치: {extras._shorten(FIRST_AID)}" for m in items]


def test_order_and_limit_are_unchanged():
    items = [msds("가성소다", ["피부 부식성/피부 자극성"]), msds("톨루엔", ["인화성 액체"]), msds("염소", ["급성 독성(흡입: 가스)"])]
    points = extras.first_aid_points(items)
    assert len(points) == extras.MAX_FIRST_AID_POINTS == 2
    assert points[0].startswith("톨루엔") and points[1].startswith("가성소다")  # 인화성 먼저, 나머지는 입력 순서
    assert "흡입했을 때" in points[0] and "눈에 들어갔을 때" in points[1]


def test_not_found_or_empty_first_aid_is_skipped():
    items = [msds("없음", ["인화성 액체"], found=False), msds("빈칸", ["인화성 액체"], first_aid=None), msds("공백", ["인화성 액체"], first_aid="  ")]
    assert extras.first_aid_points(items) == []


def test_long_inhalation_line_is_shortened_at_sentence_end():
    long_line = "흡입했을 때: 신선한 공기가 있는 곳으로 옮기고 호흡하기 쉬운 자세로 안정을 취하시오. 호흡이 힘들면 의료 훈련을 받은 사람이 산소를 공급하시오. 호흡이 멎으면 인공호흡을 하시오."
    summary = only(msds("톨루엔", ["인화성 액체"], "\n".join([EYE, long_line]))).removeprefix("톨루엔 응급조치: ")
    assert len(summary) <= extras.FIRST_AID_MAX_CHARS
    assert summary == "흡입했을 때: 신선한 공기가 있는 곳으로 옮기고 호흡하기 쉬운 자세로 안정을 취하시오."


def test_multiple_material_points_are_distinct_lines():
    points = extras.first_aid_points([msds("톨루엔", ["인화성 액체"]), msds("아세톤", ["인화성 액체"])])
    assert [p.split(" ")[0] for p in points] == ["톨루엔", "아세톤"]


def test_tbm_data_gets_the_inhalation_point():
    tbm = drafting.tbm_data([], [], msds=[msds("톨루엔", ["인화성 액체"])])
    texts = [p["point"] for p in tbm["points"]]
    assert any(t.startswith("톨루엔 응급조치: 흡입했을 때") for t in texts)


def test_through_msds_wrapper_real_shape():
    """래퍼(ghs 라벨 목록) → MsdsDto → TBM 줄. 실제 모양의 ghs 문자열에서도 인화성이 잡힌다."""
    from app.agent.msds_setup import make_msds_lookup

    class Client:
        def lookup(self, arguments):
            (term,) = arguments["materials"]
            return {"results": [{"query": term, "found": True, "name": "톨루엔", "cas_no": "108-88-3",
                                 "ghs_classes": ["인화성 액체 : 구분2|흡인 유해성 : 구분1", "자료없음"], "first_aid": FIRST_AID,
                                 "hazards": [], "regulations": [], "regulatory_tags": [], "source": "공단"}]}

    raw = make_msds_lookup(Client())({"materials": ["톨루엔"]})["results"][0]
    assert "흡입했을 때" in only(MsdsDto.model_validate(raw))


def test_unlabeled_inhalation_line_is_not_picked_over_the_first_line():
    text = "\n".join([EYE, "흡입 시 신선한 공기로 옮긴다."])  # 항목 이름(콜론)이 없는 줄은 항목으로 보지 않는다
    assert "눈에 들어갔을 때" in only(msds("톨루엔", ["인화성 액체"], text))


def test_switch_is_on_by_default():
    import importlib.util
    import pathlib

    path = pathlib.Path(extras.__file__)
    spec = importlib.util.spec_from_file_location("draft_extras_fresh", path)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)  # 시험 fixture가 건드리지 않은 새 사본
    assert fresh.PREFER_INHALATION_FIRST_AID is True
