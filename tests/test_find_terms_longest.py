"""T25: find_terms 가 긴 이름 안의 짧은 이름을 따로 잡지 않는다 (text_rules.LONGEST_TERM_FIRST).

- 스위치를 끄면 변경 전 구현과 같은 결과다 (아래 `_reference_find_terms`는 변경 전 코드를 그대로 옮긴 것).
- 켜면 변경 전 결과에서 항목을 빼기만 하고 더하지는 않는다.
"""

from __future__ import annotations

import random
import subprocess
import sys

import pytest

from app.agent import text_rules as tr
from app.agent.slots import extract_slots
from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms


# ── 변경 전 구현 (11/16 app.zip 의 text_rules.py 에서 그대로) ──
def _reference_find_word(text: str, word: str) -> int | None:
    lowered, needle = text.lower(), word.lower()
    start = 0
    while (pos := lowered.find(needle, start)) != -1:
        if needle.isascii():
            before = lowered[pos - 1] if pos > 0 else " "
            after = lowered[pos + len(needle)] if pos + len(needle) < len(lowered) else " "
            if before.isascii() and before.isalnum() or after.isascii() and after.isalnum():
                start = pos + 1
                continue
        return pos
    return None


def _reference_find_terms(text, vocab):
    found, negated = [], []
    for canonical, aliases in vocab.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pos = _reference_find_word(text, alias)
            if pos is None:
                continue
            target = negated if tr.is_negated(text, pos + len(alias)) else found
            target.append((pos, canonical))
            break
    return [c for _, c in sorted(found)], [c for _, c in sorted(negated)]


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch):
    monkeypatch.setattr(tr, "LONGEST_TERM_FIRST", True)


def mats(text):
    return find_terms(text, MATERIALS)


# ── 켰을 때 ──


def test_vocab_longer_name_hides_shorter_one():
    assert mats("메탄올 드럼") == (["메탄올"], [])
    assert mats("지하 피트에서 황화수소 확인") == (["황화수소"], [])


@pytest.mark.parametrize(
    "text",
    ["과산화수소 통", "이산화질소 누출", "일산화질소 주의", "아산화질소 통", "차아염소산 소독", "과염소산 보관", "염소산나트륨 포대",
     "이산화염소 발생", "이소프로판올 세척", "프로판올 닦기"],
)
def test_other_substances_containing_vocab_words_are_not_matched(text):
    assert mats(text) == ([], [])


def test_vocab_names_that_stand_alone_are_still_found():
    assert mats("수소 배관") == (["수소"], [])
    assert mats("질소 퍼지") == (["질소"], [])
    assert mats("염소가스 누출") == (["염소"], [])
    assert mats("프로판 용기") == (["프로판"], [])
    assert mats("메탄 보관") == (["메탄"], [])


def test_short_and_long_name_both_present_are_both_found_in_order_of_mention():
    assert mats("메탄올과 메탄 둘 다") == (["메탄올", "메탄"], [])
    assert mats("메탄 보관, 메탄올 사용") == (["메탄", "메탄올"], [])
    assert mats("과산화수소 통 옆에 수소 배관") == (["수소"], [])  # 두 번째 수소는 따로 나왔다


def test_negation_of_the_long_name_does_not_leave_the_short_name_behind():
    assert mats("메탄올은 안 씀") == ([], ["메탄올"])
    assert mats("메탄은 안 씀, 메탄올 사용") == (["메탄올"], ["메탄"])


def test_ascii_boundary_rule_is_unchanged():
    assert mats("LPGX 통") == ([], [])
    assert mats("IPA통") == (["이소프로필알코올"], [])
    assert mats("lpg 탱크 옆 ipa") == (["프로판", "이소프로필알코올"], [])


def test_equipment_is_unaffected():
    assert find_terms("이동식 크레인 사용", EQUIPMENT) == (["크레인"], [])
    assert find_terms("크레인은 안 씀, 체인블록 사용", EQUIPMENT) == (["체인블록"], ["크레인"])


def test_slot_extraction_uses_the_fix():
    slots = extract_slots("펌프실 메탄올 드럼 옆에서 용접")
    assert slots["nearby_materials"] == ["메탄올"]
    assert "nearby_materials" not in extract_slots("과산화수소 통은 지게차로 옮김")


def test_every_containing_name_covers_a_vocab_word_it_contains():
    """표에 적은 이름이 실제로 사전 단어를 품고 있어야 의미가 있다 (오타·불필요한 항목을 잡는다)."""
    aliases = [a for al in MATERIALS.values() for a in al]
    for name in tr.CONTAINING_NAMES:
        inside = [a for a in aliases if a != name and a in name]
        assert inside, f"{name} 은(는) 사전 단어를 품지 않는다"
        assert name not in aliases


# ── 껐을 때: 변경 전과 같다 ──


def test_off_switch_gives_the_old_result(monkeypatch):
    monkeypatch.setattr(tr, "LONGEST_TERM_FIRST", False)
    assert mats("메탄올 드럼") == (["메탄", "메탄올"], [])
    assert mats("과산화수소 통") == (["수소"], [])
    assert mats("메탄올은 안 씀") == (["메탄"], ["메탄올"])


def _corpus() -> list[str]:
    from scripts.plan_eval.harness import DEFAULT_EVAL_SET, load_cases

    texts = [c.text for c in load_cases(DEFAULT_EVAL_SET)]
    words = [a for v in (*MATERIALS.values(), *EQUIPMENT.values()) for a in v] + list(tr.CONTAINING_NAMES)
    fillers = ["", "은 안 씀", "을 사용", "통", " 옆에", "없이", "가스", "올", "산"]
    rng = random.Random(1117)
    for _ in range(600):
        parts = []
        for _ in range(rng.randint(1, 5)):
            parts.append(rng.choice(words) + rng.choice(fillers))
        texts.append(rng.choice([" ", ", ", ". ", "\n"]).join(parts))
    return texts


def test_off_switch_equals_reference_on_a_large_corpus(monkeypatch):
    monkeypatch.setattr(tr, "LONGEST_TERM_FIRST", False)
    for text in _corpus():
        for vocab in (MATERIALS, EQUIPMENT):
            assert find_terms(text, vocab) == _reference_find_terms(text, vocab), text


def _compound_words() -> list[str]:
    """사전 표기나 CONTAINING_NAMES 중 다른 사전 표기를 품은 것 (이것이 글에 없으면 결과가 달라질 수 없다)."""
    words = [a for v in MATERIALS.values() for a in v] + list(tr.CONTAINING_NAMES)
    aliases = [a for v in MATERIALS.values() for a in v]
    return [w for w in words if any(a != w and a.lower() in w.lower() for a in aliases)]


def test_result_changes_only_when_a_longer_name_is_in_the_text():
    compound = [w.lower() for w in _compound_words()]
    changed = 0
    for text in _corpus():
        for vocab in (MATERIALS, EQUIPMENT):
            ref = _reference_find_terms(text, vocab)
            got = find_terms(text, vocab)
            if ref != got:
                changed += 1
                assert any(w in text.lower() for w in compound), text
        if not any(w in text.lower() for w in compound):
            assert find_terms(text, MATERIALS) == _reference_find_terms(text, MATERIALS), text
    assert changed > 0  # 말뭉치가 이 경우를 실제로 건드린다


def test_a_real_mention_after_a_negated_long_name_is_found():
    # 변경 전에는 첫 자리(산화염소 안)만 보고 '염소'를 부정으로 처리해 뒤의 진짜 염소를 놓쳤다
    assert mats("산화염소는 안 씀, 염소통 옆") == (["염소"], [])
    assert mats("메탄올은 안 씀, 메탄 사용") == (["메탄"], ["메탄올"])


def test_eval_set_differences_are_exactly_the_four_t25_cases(monkeypatch):
    from scripts.plan_eval.harness import DEFAULT_EVAL_SET, load_cases

    diffs = {}
    for case in load_cases(DEFAULT_EVAL_SET):
        ref = _reference_find_terms(case.text, MATERIALS)
        now = find_terms(case.text, MATERIALS)
        if ref != now:
            diffs[case.id] = (ref[0], now[0])
        assert find_terms(case.text, EQUIPMENT) == _reference_find_terms(case.text, EQUIPMENT)
    assert diffs == {
        "p05": (["황화수소", "수소"], ["황화수소"]),
        "n01": (["메탄", "메탄올"], ["메탄올"]),
        "o03": (["수소"], []),
        "m02": (["메탄", "메탄올", "에탄올"], ["메탄올", "에탄올"]),
    }


# ── 사전 모양과 상관없이 ──


def test_any_alias_of_any_entry_can_cover_a_shorter_name():
    vocab = {"긴것": ("가나", "가나다라"), "짧은것": ("나다",)}
    assert find_terms("가나다라 보관", vocab) == (["긴것"], [])  # 둘째 표기(가나다라)가 나다를 덮는다
    assert find_terms("나다만 있음", vocab) == (["짧은것"], [])


def test_a_later_uncovered_occurrence_right_next_to_a_covered_one_is_found():
    vocab = {"S": ("가가",), "L": ("나가가",)}
    # 가가 는 1번 자리(나가가 안)와 2번 자리(나가가 밖으로 걸침)에 있다. 두 번째가 진짜 언급이다.
    assert find_terms("나가가가", vocab) == (["L", "S"], [])


def test_defaults_are_on_in_a_fresh_interpreter():
    code = (
        "from app.agent import text_rules as t;"
        "print(t.LONGEST_TERM_FIRST, t.find_terms('메탄올 드럼', t.MATERIALS)[0])"
    )
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-300:]
    assert out.stdout.strip() == "True ['메탄올']"
