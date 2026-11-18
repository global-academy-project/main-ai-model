"""T30 (11/17): 사전 확장(사다리차·황산·염산·질산·초산) · 금지 부정('사용 금지'·'쓸 일 없') · 요약 단어 가드.

현장 문장 측정에서 규칙이 놓친 말을 더한 것이다. 슬롯 사전은 주변 물질·투입 장비 줄, 작업 분류, 계획 단계 검증이 함께 쓰므로
번지는 범위를 같이 시험한다. 스위치(EXTENDED_VOCAB · PROHIBIT_NEGATION · SUMMARY_WORD_GUARD)를 끄면 변경 전과 같아야 한다.
검증 범위: 코드 실행 (모델·서버는 부르지 않는다).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("app.agent.work_register")

from app.agent import plan_brief as pb  # noqa: E402
from app.agent import text_rules as tr  # noqa: E402
from app.agent.classify import classify_text  # noqa: E402
from app.agent.evidence_collector import WorkContext  # noqa: E402
from app.agent.evidence_collector.vocab import DEFAULT_CORPORA  # noqa: E402
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms, is_negated  # noqa: E402
from app.llm.llm_base import LLMClient  # noqa: E402
from app.llm.llm_types import LLMResponse  # noqa: E402

VOCAB = {**EQUIPMENT, **MATERIALS}


def _namespace(**replace):
    """text_rules 소스의 스위치만 바꿔 새 이름공간에서 실행한다 (다른 모듈에 영향 없음)."""
    source = Path(tr.__file__).read_text(encoding="utf-8")
    for old, new in replace.items():
        assert f"{old} = True" in source, old
        source = source.replace(f"{old} = True", f"{old} = {new}")
    ns: dict = {"__name__": "text_rules_variant", "__file__": tr.__file__}
    exec(compile(source, "text_rules_variant", "exec"), ns)  # noqa: S102 - 시험: 스위치만 바꿔 실행
    return ns


# ---------------------------------------------------------------- 사전 확장


def test_ladder_truck_is_its_own_equipment_not_a_ladder():
    assert find_terms("굴뚝 청소 작업. 사다리차 사용.", EQUIPMENT)[0] == ["사다리차"]
    assert find_terms("사다리 올라가서 점검", EQUIPMENT)[0] == ["사다리"]
    assert sorted(find_terms("사다리차와 사다리 둘 다 사용", EQUIPMENT)[0]) == ["사다리", "사다리차"]


@pytest.mark.parametrize("name", ["황산", "염산", "질산", "초산"])
def test_new_chemicals_are_in_the_dictionary(name):
    assert find_terms(f"{name} 탱크 옆에서 작업", MATERIALS)[0] == [name]
    assert find_terms(f"{name}은 안 씀", MATERIALS) == ([], [name])


def test_acetic_acid_is_the_same_item_as_chosan():
    assert find_terms("아세트산 희석", MATERIALS)[0] == ["초산"]


@pytest.mark.parametrize("text", [
    "아황산가스 측정", "황산구리 용액 준비", "황산나트륨 포대", "황산암모늄 비료", "황산염 처리", "과황산 용액",
    "초산에틸 도료", "초산비닐 접착제", "아질산 소다", "질산암모늄 보관", "질산칼륨 포대", "염산염 시약",
])
def test_names_that_contain_a_new_chemical_are_not_that_chemical(text):
    assert find_terms(text, MATERIALS) == ([], []), text


def test_old_dictionary_words_still_work_around_the_new_ones():
    assert find_terms("염소와 염산 둘 다 있음", MATERIALS)[0] == ["염소", "염산"] or sorted(find_terms("염소와 염산 둘 다 있음", MATERIALS)[0]) == ["염산", "염소"]
    assert find_terms("과염소산 용액", MATERIALS) == ([], [])  # 염소 아님 (T25)
    assert find_terms("황화수소 측정", MATERIALS)[0] == ["황화수소"]  # 황산 아님, 수소 아님


def test_extended_vocab_off_is_exactly_the_old_dictionary():
    off = _namespace(EXTENDED_VOCAB=False)
    assert "사다리차" not in off["EQUIPMENT"]
    for name in ("황산", "염산", "질산", "초산"):
        assert name not in off["MATERIALS"]
    assert off["CONTAINING_NAMES"] == ("과산화수소", "산화질소", "염소산", "산화염소", "프로판올")
    assert list(off["EQUIPMENT"]) == [k for k in EQUIPMENT if k != "사다리차"]
    assert list(off["MATERIALS"]) == [k for k in MATERIALS if k not in ("황산", "염산", "질산", "초산")]
    assert off["find_terms"]("사다리차 사용", off["EQUIPMENT"])[0] == ["사다리"]  # 옛 동작: 사다리로 잡힌다


# ---------------------------------------------------------------- 금지 부정


NEGATED = [
    "그라인더는 사용 금지 구역이라 수공구로 작업.", "그라인더 사용 금지.", "용접기 쓸 일 없어서 볼트 체결로만 진행함.", "토치 쓸 일 없음.",
    "토치 쓸 계획 없음.", "토치 쓸 계획이 없음.", "크레인은 쓸 필요가 없음.", "드릴 쓸 필요 없음.", "아세톤 사용 금지", "시너는 쓸 일이 없습니다.",
]
KEPT = [
    "그라인더 사용 금지 해제 후 연마.", "그라인더 사용 금지 표지 부착 작업.", "그라인더 사용 금지 표시 확인.", "크레인 사용 금지 해제됨.",
    "토치 쓸 일 생기면 호출.", "크레인 쓸 계획 세움.", "드릴 쓸 필요 있음.", "그라인더 사용해서 연마, 금지 구역 아님.", "크레인 사용 후 금지 표지 제거.",
    "그라인더로 연마. 사용 금지 구역은 따로 표시함.",
]


@pytest.mark.parametrize("text", NEGATED)
def test_prohibit_and_not_needed_forms_are_negated(text):
    found, negated = find_terms(text, VOCAB)
    assert found == [] and len(negated) == 1, (text, found, negated)


@pytest.mark.parametrize("text", KEPT)
def test_lifted_or_unrelated_prohibitions_are_not_negated(text):
    found, _ = find_terms(text, VOCAB)
    assert len(found) == 1, (text, found)


def test_negation_does_not_cross_lines():
    found, negated = find_terms("그라인더로 연마\n사용 금지 구역 표시", VOCAB)
    assert found == ["그라인더"] and negated == []


def test_prohibit_switch_off_is_exactly_the_polite_only_pattern():
    off = _namespace(PROHIBIT_NEGATION=False)
    assert off["_PROHIBIT_WORDS"] == ""
    assert tr.NEGATION_AFTER.pattern == off["NEGATION_AFTER"].pattern.removesuffix(")") + tr._PROHIBIT_WORDS + ")"  # noqa: SLF001
    for text in NEGATED:
        assert off["find_terms"](text, {**off["EQUIPMENT"], **off["MATERIALS"]})[0] != [], text  # 끄면 다시 부정이 아니다
    assert off["find_terms"]("크레인은 안 씀.", off["EQUIPMENT"])[0] == []  # 옛 부정은 그대로


def test_is_negated_sees_the_new_forms_right_after_the_word():
    assert is_negated("그라인더 사용 금지", len("그라인더"))
    assert not is_negated("그라인더 사용 금지 해제", len("그라인더"))


# ---------------------------------------------------------------- 번지는 범위: 작업 분류·슬롯·계획 검증


def test_classification_still_sees_height_work_for_ladder_truck():
    """사다리차는 '사다리'를 품은 말이라 고소 키워드에 그대로 걸린다 (사전에서 따로 잡혀도 분류는 같다)."""
    assert "고소" in classify_text("굴뚝 청소 작업. 사다리차 사용.").work_types


def test_classification_ignores_a_prohibited_tool_like_other_negations():
    """'그라인더 사용 금지'는 그라인더를 쓰지 않는다는 말이라 화기로 보지 않는다 ('그라인더는 안 씀'과 같다)."""
    assert "화기" not in classify_text("배관 절단. 그라인더 사용 금지 구역이라 수공구로 작업.").work_types
    assert "화기" in classify_text("배관 절단. 그라인더 사용 금지 해제 후 작업.").work_types  # 풀린 금지는 쓴다는 뜻


def test_plan_validation_follows_the_new_negations():
    text = "펌프실 베어링 교환\n그라인더는 사용 금지 구역이라 수공구로 작업.\n용접기 쓸 일 없음.\n드릴로 구멍 뚫음"
    assert not pb.mentioned(text, "그라인더") and not pb.mentioned(text, "용접기")
    kept, dropped = pb.clean_equipment(["그라인더", "용접기", "드릴"], text)
    assert kept == ["드릴"] and dropped == 2


def test_plan_validation_knows_the_new_chemicals_as_dictionary_names():
    kept, dropped = pb.clean_materials(["황산", "아세트산"], "황산 탱크 옆에서 작업. 아세트산 희석")
    assert kept == ["황산", "초산"] and dropped == 0  # 사전 표기는 표준 이름으로


# ---------------------------------------------------------------- 요약 단어 가드


TEXT = "R-201 반응기 하부 배관 플랜지 볼트 교체. 고착되면 그라인더로 절단. 3명, 15시까지."


def test_unsupported_words_finds_only_words_not_in_the_text():
    assert pb.unsupported_summary_words("R-201 반응기 하부 배관 플랜지 볼트 교체", TEXT) == []
    assert pb.unsupported_summary_words("R-201 반응기 하부 배관 플랜지 볼트 교체함", TEXT) == []  # 끝 한 글자는 떼고 본다 (교체함 → 교체)
    assert pb.unsupported_summary_words("반응기 하부 랜치 교체", TEXT) == ["랜치"]
    assert pb.unsupported_summary_words("반응기 하부 배관 점검 및 교체", TEXT) == ["점검"]


def test_unsupported_words_is_the_same_rule_as_the_eval_table():
    from scripts.plan_eval.harness import summary_support

    for summary in ("반응기 하부 랜치 교체", "R-201 배관 플랜지 교체함", "하부 배관 조임 작업", "펌프실 베어링 교환"):
        assert pb.unsupported_summary_words(summary, TEXT) == summary_support(summary, TEXT)[1], summary


def test_one_letter_words_and_digits_follow_the_same_policy_as_eval():
    assert pb.unsupported_summary_words("3명 15시", TEXT) == []
    assert pb.unsupported_summary_words("가 나 다", TEXT) == []  # 한 글자는 보지 않는다


class _Stub(LLMClient):
    name = "STUB"

    def __init__(self, text: str) -> None:
        super().__init__(None)
        self.text = text

    def chat(self, messages, *, system="", tools=None, max_tokens=None):
        return LLMResponse(text=self.text, stop_reason="end")


def _plan(summary: str, *, guard: bool = True):
    llm = _Stub(json.dumps({"summary": summary, "equipment": ["그라인더"], "materials": []}, ensure_ascii=False))
    work = WorkContext(text=TEXT, work_types=["화기"], nearby_materials=[], equipment=[])
    original = pb.SUMMARY_WORD_GUARD
    pb.SUMMARY_WORD_GUARD = guard
    try:
        return pb.BriefPlanner(llm=llm, corpora=DEFAULT_CORPORA).plan(work)
    finally:
        pb.SUMMARY_WORD_GUARD = original


def test_planner_uses_the_model_summary_when_every_word_has_support():
    plan = _plan("R-201 반응기 하부 배관 플랜지 볼트 교체")
    assert plan.summary == "R-201 반응기 하부 배관 플랜지 볼트 교체" and plan.source == "llm"


def test_planner_replaces_a_summary_with_an_unsupported_word_by_the_rule_summary():
    plan = _plan("R-201 반응기 하부 배관 랜치 교체")
    assert plan.source == "fallback"
    assert plan.summary == pb.one_line(TEXT, pb.BRIEF_SUMMARY_CHARS * 2)
    assert plan.equipment == ["그라인더"]  # 요약만 바뀌고 공구·물질 후보는 그대로


def test_planner_guard_off_keeps_the_old_behavior():
    plan = _plan("R-201 반응기 하부 배관 랜치 교체", guard=False)
    assert plan.summary == "R-201 반응기 하부 배관 랜치 교체" and plan.source == "llm"


def test_summary_guard_is_on_by_default():
    assert pb.SUMMARY_WORD_GUARD is True and tr.EXTENDED_VOCAB is True and tr.PROHIBIT_NEGATION is True
