"""규칙 판정 제목 한글 보강(RULE_TITLE_FILL). 모양은 11/16 TBM 화면(#2 밀폐·#3 화기+고소)을 따른다."""

from __future__ import annotations

import pytest

from app.agent import drafting, rule_titles
from app.agent.rule_titles import GENERIC_TITLE, RULE_TITLES, blocking_titles, display_title
from app.agent.work_register import _block_titles
from app.schemas.agent_work import RuleCheckDto, RuleEvaluationDto

C_TITLES = {  # C의 check_rules.py TITLES 값 (한글로 이미 나오는 것)
    "CONFINED_O2_RANGE": "산소 농도",
    "CONFINED_CO2_MAX": "탄산가스 농도",
    "CONFINED_CO_MAX": "일산화탄소 농도",
    "CONFINED_H2S_MAX": "황화수소 농도",
    "CONFINED_LEL_MAX": "가연성 가스 농도",
    "HOT_LEL_MAX": "가연성 가스 농도",
    "HOT_FIRE_WATCHER": "화재감시자 지정",
    "HOT_EXTINGUISHER": "소화기 비치",
    "HOT_SPARK_PREVENTION": "불티 비산 방지",
    "HOT_FLAMMABLE_VAPOR_REMOVED": "인화성 증기 제거",
}


@pytest.fixture(autouse=True)
def switch_on(monkeypatch):
    monkeypatch.setattr(rule_titles, "RULE_TITLE_FILL", True)


def ev(code: str, title: str | None = None, *, status: str = "MISSING", severity: str = "BLOCK") -> RuleEvaluationDto:
    """C의 도구가 내는 모양: 표에 없는 코드는 title == rule_code."""
    return RuleEvaluationDto(rule_code=code, title=C_TITLES.get(code, code) if title is None else title,
                             status=status, severity=severity, message="m", basis="b")


def check(*evaluations: RuleEvaluationDto) -> RuleCheckDto:
    return RuleCheckDto(blocked=True, evaluations=list(evaluations))


def sentence(rule_check: RuleCheckDto) -> str:
    """TBM 첫 요점 (화면 문장)."""
    titles = _block_titles(rule_check, ["차단 사유"])
    return drafting.tbm_data([], titles)["points"][0]["point"]


def test_six_missing_codes_get_korean_names():
    assert RULE_TITLES == {
        "CONFINED_WATCHER": "감시인 지정", "CONFINED_VENTILATION": "작업 전 환기", "CONFINED_RESCUE_EQUIPMENT": "대피·구출용 기구 비치",
        "HEIGHT_FALL_PROTECTION": "추락 방지 조치", "HEIGHT_HELMET": "안전모 착용", "HEIGHT_HARNESS": "안전대 착용",
    }
    for code, name in RULE_TITLES.items():
        assert display_title(code, code) == name


def test_screen_tbm3_hot_and_height_has_no_english():
    rule_check = check(
        ev("HOT_EXTINGUISHER"), ev("HOT_FIRE_WATCHER"), ev("HOT_FLAMMABLE_VAPOR_REMOVED"), ev("HOT_LEL_MAX"), ev("HOT_SPARK_PREVENTION"),
        ev("HEIGHT_FALL_PROTECTION"), ev("HEIGHT_HARNESS"), ev("HEIGHT_HELMET"),
    )
    assert sentence(rule_check) == (
        "소화기 비치, 화재감시자 지정, 인화성 증기 제거, 가연성 가스 농도, 불티 비산 방지, 추락 방지 조치, 안전대 착용, 안전모 착용 전에는 작업을 시작하지 않는다"
    )


def test_screen_tbm2_confined_has_no_english():
    rule_check = check(
        ev("CONFINED_CO2_MAX"), ev("CONFINED_CO_MAX"), ev("CONFINED_H2S_MAX"), ev("CONFINED_LEL_MAX"), ev("CONFINED_O2_RANGE"),
        ev("CONFINED_RESCUE_EQUIPMENT"), ev("CONFINED_VENTILATION"), ev("CONFINED_WATCHER"),
    )
    assert sentence(rule_check) == (
        "탄산가스 농도, 일산화탄소 농도, 황화수소 농도, 가연성 가스 농도, 산소 농도, 대피·구출용 기구 비치, 작업 전 환기, 감시인 지정 전에는 작업을 시작하지 않는다"
    )


def test_korean_title_from_c_wins_over_the_table():
    # C가 TITLES에 추가한 뒤에는 C의 제목을 그대로 쓴다 (이 표는 쓰이지 않는다)
    assert display_title("HEIGHT_HELMET", "안전모 착용 확인") == "안전모 착용 확인"
    assert blocking_titles([ev("HEIGHT_HELMET", "머리 보호구")]) == ["머리 보호구"]


def test_unknown_english_code_becomes_generic_once():
    titles = blocking_titles([ev("NEW_RULE_A"), ev("NEW_RULE_B"), ev("HOT_EXTINGUISHER")])
    assert titles == [GENERIC_TITLE, "소화기 비치"]
    assert blocking_titles([ev("NEW_RULE_A"), ev("HOT_EXTINGUISHER"), ev("NEW_RULE_B")]) == [GENERIC_TITLE, "소화기 비치"]
    assert "NEW_RULE" not in sentence(check(ev("NEW_RULE_A")))


def test_only_blocking_unpassed_items_are_listed():
    rule_check = check(
        ev("HEIGHT_HELMET", status="PASS"), ev("HEIGHT_HARNESS", severity="WARN"), ev("HEIGHT_FALL_PROTECTION"), ev("NEW_RULE_A", status="PASS"),
    )
    assert blocking_titles(rule_check.evaluations) == ["추락 방지 조치"]


def test_same_korean_title_twice_is_kept_as_before():
    # 화기+밀폐 둘 다일 때 "가연성 가스 농도"가 두 번 나오는 예전 동작은 바꾸지 않는다
    assert blocking_titles([ev("HOT_LEL_MAX"), ev("CONFINED_LEL_MAX")]) == ["가연성 가스 농도", "가연성 가스 농도"]


def test_no_block_reasons_or_no_rule_check_keeps_old_fallback():
    assert _block_titles(None, ["x"]) == ["규칙 판정"]
    assert _block_titles(check(ev("HEIGHT_HELMET")), []) == []
    assert _block_titles(check(ev("HOT_EXTINGUISHER", status="PASS")), ["x"]) == ["규칙 판정"]


def test_switch_off_is_exactly_the_old_behavior(monkeypatch):
    monkeypatch.setattr(rule_titles, "RULE_TITLE_FILL", False)
    rule_check = check(ev("HOT_EXTINGUISHER"), ev("HEIGHT_HELMET"), ev("NEW_RULE_A"), ev("NEW_RULE_B"))
    old = [e.title for e in rule_check.evaluations if e.severity == "BLOCK" and e.status != "PASS"]  # 예전 한 줄 그대로
    assert old == ["소화기 비치", "HEIGHT_HELMET", "NEW_RULE_A", "NEW_RULE_B"]
    assert blocking_titles(rule_check.evaluations) == old
    assert _block_titles(rule_check, ["x"]) == old
    assert display_title("HEIGHT_HELMET", "HEIGHT_HELMET") == "HEIGHT_HELMET"


def test_table_values_are_korean_and_unique():
    for name in RULE_TITLES.values():
        assert any("가" <= ch <= "힣" for ch in name)
    assert len(set(RULE_TITLES.values())) == len(RULE_TITLES)


def test_shipped_default_is_on():
    # 위 autouse 고정과 상관없이 파일에 적힌 기본값을 본다
    import importlib.util
    spec = importlib.util.spec_from_file_location("rule_titles_fresh", rule_titles.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.RULE_TITLE_FILL is True


def test_generic_wording_is_the_existing_fallback_text():
    assert GENERIC_TITLE == "규칙 판정"  # work_register의 예전 대체 문구와 같다


def test_code_is_looked_up_even_when_title_differs_from_code():
    assert display_title("HEIGHT_HELMET", "Helmet worn") == "안전모 착용"
    assert display_title("SOMETHING_NEW", "Helmet worn") == GENERIC_TITLE
    assert blocking_titles([ev("HEIGHT_HARNESS", "Harness")]) == ["안전대 착용"]


def test_switch_off_keeps_repeated_titles(monkeypatch):
    monkeypatch.setattr(rule_titles, "RULE_TITLE_FILL", False)
    assert blocking_titles([ev("A", GENERIC_TITLE), ev("B", GENERIC_TITLE)]) == [GENERIC_TITLE, GENERIC_TITLE]
