"""사용자가 쓴 5문장 + 어렵게 만든 22문장 (scripts/plan_eval/eval_set_user_hard.jsonl) 검사 (T28 11/17).

이 세트는 합성 학습 데이터와 별개로 현장 문체·오타·약어·규칙의 빈틈을 보려는 것이다. 정답은 내가 정한 임시값이다.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.plan_eval import harness

DIR = Path(__file__).resolve().parents[1] / "scripts" / "plan_eval"
HARD = DIR / "eval_set_user_hard.jsonl"
USER_TEXTS = [
    "8시부터 1시까지 창고에서 3명 용접 작업. 용접기 사용하고, 메탄올 존재해.",
    "11시부터 5시까지 3번 플랜트 굴뚝 청소 작업. 사다리차 사용. 황화수소 측정 전.",
    "2번 창고 페인트칠, 2명이서 시너 사용.",
    "5번 플랜트 화합물 이동 작업, 황산 옮길거야.",
    "지붕 보수 작업. 크레인 사용함.",
]
# 지금 슬롯 사전(규칙)이 틀리게 더하는 문장: 최종 = 슬롯 + 모델이라 모델이 정답을 내도 최종값은 틀린다 (규칙을 고치기 전까지의 상한)
# 11/17(T30): 사다리차·'사용 금지'·'쓸 일 없'을 규칙에 더해 u02·h05·h06 은 풀렸다. 남은 h09 는 '~로 바꿈'(앞 줄에서 하려던 것을 뒷 줄에서 바꿈)이라 규칙으로는 못 푼다
KNOWN_RULE_GAPS = {"h09": ["그라인더"]}


def test_loads_with_27_cases_and_user_sentences_are_verbatim():
    cases = harness.load_cases(HARD)
    assert len(cases) == 27
    texts = {c.id: c.text for c in cases}
    assert [texts[f"u0{i}"] for i in range(1, 6)] == USER_TEXTS


def test_ids_never_collide_with_the_other_eval_sets():
    ids = [c.id for name in ("eval_set.jsonl", "eval_set_multiline.jsonl", "eval_set_user_hard.jsonl") for c in harness.load_cases(DIR / name)]
    assert len(ids) == 79 and len(set(ids)) == 79


def test_file_is_lf_and_every_line_is_json():
    raw = HARD.read_bytes()
    assert b"\r" not in raw
    for line in raw.decode("utf-8").splitlines():
        json.loads(line)


def test_every_kind_is_used_and_none_cases_have_empty_gold():
    cases = harness.load_cases(HARD)
    assert {c.kind for c in cases} == set(harness.KINDS)
    for c in cases:
        if c.kind in ("none", "distractor"):
            assert not c.gold_equipment and not c.gold_materials, c.id


def test_known_rule_gaps_are_exactly_the_cases_where_the_slot_adds_a_wrong_item():
    """슬롯 사전이 정답에 없는 말을 더하는 문장이 이 1건뿐인지 센다. 규칙을 고치면 이 목록이 줄어야 하고, 그때 이 시험과 README 를 같이 고친다."""
    found: dict[str, list[str]] = {}
    for c in harness.load_cases(HARD):
        eq, mat = harness.slot_terms(c.text)
        gold = {*harness._canon(c.gold_equipment, harness.pb._EQUIPMENT_KEYS), *harness._canon(c.gold_materials, harness.pb._MATERIAL_KEYS)}  # noqa: SLF001
        extra = [x for x in (*eq, *mat) if x not in gold]
        if extra:
            found[c.id] = extra
    assert found == KNOWN_RULE_GAPS
