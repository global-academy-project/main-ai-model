"""T28 3번: 정중형·완료형 부정('안 씁니다', '안 썼음', '쓰지 않음', '필요하지 않음', '투입하지 않음')을 부정으로 본다 (text_rules.POLITE_NEGATION).

부정 규칙은 슬롯(주변 물질·투입 장비), 작업 분류, 계획 단계의 검증이 함께 쓰므로 한 곳을 고치면 셋에 번진다. 번지는 범위도 시험으로 고정한다.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from app.agent import plan_brief as pb
from app.agent import text_rules as tr
from app.agent.classify import classify_text
from app.agent.slots import merge_slots
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms, is_negated

NEGATED = [
    "그라인더는 안 씁니다.", "그라인더는 안 썼음.", "그라인더는 안 썼어요.", "크레인은 쓰지 않음.", "토치는 필요하지 않음.", "용접기는 투입하지 않음.",
    "드릴은 쓰지 않기로 함.", "발전기 필요치 않음.", "그라인더 안 합니다", "그라인더는 안 했음", "송풍기는 가동하지 않음", "지게차는 동원하지 않음",
    "아세톤은 사용하지 않습니다.", "아세톤은 쓰지 않음",
]
KEPT = [
    "크레인으로 작업 안전하게 진행.", "그라인더 안전 점검 후 사용.", "토치 사용해서 가열.", "크레인 점검하지 않으면 안 됨.", "용접기 사용 전 확인.",
    "사다리 안정 확보.", "그라인더로 연마. 안 쓰는 공구는 치움.", "드릴 작업 중 안내 방송.", "크레인 작업하지 않는 날은 쉼.", "가스측정기 합격 확인.",
    "비계 안쪽 점검.", "그라인더 쓰기 편함.", "크레인 필요하면 호출.", "지게차 사용해서 운반.", "발전기 가동 후 작업.",
]
VOCAB = {**EQUIPMENT, **MATERIALS}


@pytest.mark.parametrize("text", NEGATED)
def test_polite_negations_are_negated(text):
    found, negated = find_terms(text, VOCAB)
    assert found == [] and len(negated) == 1, (text, found, negated)


@pytest.mark.parametrize("text", KEPT)
def test_ordinary_sentences_are_not_negated(text):
    found, negated = find_terms(text, VOCAB)
    assert negated == [], (text, negated)


def test_is_negated_looks_only_right_after_the_word_on_the_same_line():
    assert is_negated("그라인더는 안 씁니다", len("그라인더"))
    assert not is_negated("그라인더\n안 씁니다", len("그라인더"))  # 줄바꿈을 넘지 않는다 (앞 발화의 단어를 다음 발화가 부정하지 않게)
    assert not is_negated("그라인더로 연마. 안 씁니다", len("그라인더"))


def test_rules_that_were_already_negations_still_are():
    for text in ["크레인은 안 씀.", "크레인 사용 안 함.", "크레인 사용하지 않음.", "크레인 없이 진행.", "크레인 제외", "크레인 미사용"]:
        assert find_terms(text, VOCAB)[0] == [], text
    for text in ["크레인 말고 호이스트.", "크레인 대신 호이스트"]:
        assert find_terms(text, VOCAB)[0] == ["호이스트"], text  # 바꿔 쓴 것만 남는다
    assert find_terms("크레인 말고도 호이스트 사용", VOCAB)[0] == ["크레인", "호이스트"]  # '말고도'는 부정이 아니다 (기존 규칙)


# ---------------------------------------------------------------- 번지는 곳


def _fold(*messages):
    slots = None
    for message in messages:
        slots = merge_slots(slots, message, today=date(2026, 1, 15))
    return slots


def test_slots_drop_equipment_withdrawn_politely_and_do_not_add_it():
    assert _fold("도장 부스 점검. 그라인더는 안 씁니다.").equipment == []
    assert _fold("그라인더로 연마하려 했음.", "그라인더는 쓰지 않기로 함.").equipment == []  # 뒤 발화의 거둠이 앞에서 잡힌 장비를 뺀다
    assert _fold("토치와 용접기 준비.", "토치는 필요하지 않음.").equipment == ["용접기"]
    assert _fold("아세톤은 사용하지 않습니다. 에탄올 있음.").nearby_materials == ["에탄올"]


def test_hot_work_kind_and_classification_follow_the_same_rule():
    """작업 분류도 같은 규칙을 쓴다: '크레인은 쓰지 않음'은 '크레인은 안 씀'과 똑같이 취급된다 (예전에는 정중형이면 중량물로 분류됐다)."""
    same = classify_text("배관 교체. 크레인은 안 씀.")
    polite = classify_text("배관 교체. 크레인은 쓰지 않음.")
    assert polite.work_types == same.work_types and "중량물" not in polite.work_types
    assert "중량물" in classify_text("배관 교체. 크레인으로 인양.").work_types  # 부정이 없으면 그대로


def test_plan_validation_treats_polite_negation_as_not_mentioned():
    text = "펌프실 베어링 교환\n그라인더는 안 씁니다.\n토치는 필요하지 않음."
    assert not pb.mentioned(text, "그라인더") and not pb.mentioned(text, "토치")
    kept, dropped = pb.clean_equipment(["그라인더", "토치", "드릴"], text + "\n드릴로 구멍 뚫음")
    assert kept == ["드릴"] and dropped == 2


# ---------------------------------------------------------------- 스위치


def _namespace(**replace):
    source = Path(tr.__file__).read_text(encoding="utf-8")
    for old, new in replace.items():
        assert f"{old} = True" in source
        source = source.replace(f"{old} = True", f"{old} = {new}")
    ns: dict = {"__name__": "text_rules_variant", "__file__": tr.__file__}
    exec(compile(source, "text_rules_variant", "exec"), ns)  # noqa: S102 - 시험: 소스의 스위치만 바꿔 새 이름공간에서 실행한다 (다른 모듈에 영향 없음)
    return ns


def test_switch_off_is_exactly_the_old_pattern():
    off = _namespace(POLITE_NEGATION=False, PROHIBIT_NEGATION=False)  # 금지 부정(T30)은 따로 끈다: 이 시험은 정중형 스위치만 본다
    old = (r"^[ \t]*(?:은|는|도|을|를|이|가)?[ \t]*(?:"
           r"안\s*(?:씀|써|쓴|쓸|쓰|함|해|한)|사용\s*(?:안|하지|않|없)|미사용|없음|없이|필요\s*없|제외" r"|말고(?!도)|대신|아니라|아니고" r")")
    assert off["NEGATION_AFTER"].pattern == old
    assert off["_POLITE_WORDS"] == ""
    for text in NEGATED[:3]:
        assert off["find_terms"](text, VOCAB)[0] != [], text  # 끄면 정중형은 다시 부정이 아니다
    for text in ["크레인은 안 씀.", "크레인 사용하지 않음."]:
        assert off["find_terms"](text, VOCAB)[0] == [], text
    assert off["find_terms"]("크레인 말고 호이스트", VOCAB)[0] == ["호이스트"]


def test_switch_on_is_the_default_and_only_adds_alternatives():
    assert tr.POLITE_NEGATION is True
    off = _namespace(POLITE_NEGATION=False, PROHIBIT_NEGATION=False)["NEGATION_AFTER"].pattern
    on = _namespace(PROHIBIT_NEGATION=False)["NEGATION_AFTER"].pattern
    assert on.startswith(off[:-1]) and on.endswith(")")  # 끝 괄호만 닫지 않은 채 대안이 이어 붙는다
    extra = on[len(off) - 1 : -1]
    assert extra.startswith("|") and "씁" in extra and "투입하" in extra


def test_polite_words_do_not_widen_to_arbitrary_verbs():
    """'~하지 않' 전체로 넓히지 않았다: 장비 이름 바로 뒤의 다른 동사는 부정으로 보지 않는다."""
    for verb in ["점검하", "교체하", "확인하", "설치하", "연결하"]:
        assert find_terms(f"크레인은 {verb}지 않으면 위험", VOCAB)[0] == ["크레인"], verb
    assert not re.search(r"\(\?:[^)]*점검", tr._POLITE_WORDS)


def test_text_rules_keeps_crlf_line_endings():
    raw = Path(tr.__file__).read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n") and raw.count(b"\r\n") > 100
