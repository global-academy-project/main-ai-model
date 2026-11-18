"""scripts/make_plan_train.py 시험: 학습 데이터가 서버와 같은 입력·검증을 쓰는지, 평가 문장과 겹치지 않는지, 검사가 실제로 거르는지."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from app.agent import plan_brief as pb
from app.agent.evidence_collector.models import WorkContext
from app.agent.plan_adapter import ADAPTER_SYSTEM_PROMPT, build_adapter_input
from app.agent.slots import full_text
from scripts import make_plan_train as mpt
from scripts.plan_eval.harness import DEFAULT_EVAL_SET, load_cases

SEED = 2026


@pytest.fixture(scope="module")
def guard():
    return mpt.EvalGuard.load()


@pytest.fixture(scope="module")
def rows(guard):
    return mpt.generate(700, SEED, guard)


@pytest.fixture(scope="module")
def valid_rows(guard):
    return mpt.generate(150, SEED + 1, guard, holdout=True)


def piece(role="task", text="", **kw):
    return mpt.Piece(role, text, **kw)


def run_check(utterances, *, summary, equipment=(), materials=(), pieces=None):
    """정답을 직접 지정해 check_row 를 부른다 (잘못된 정답을 거르는지 보려고)."""
    slots = mpt.slots_for(list(utterances))
    sample = mpt.Sample("plain", pieces or [piece(text=utterances[0])], summary)
    mpt.check_row(sample, list(utterances), slots, summary, list(equipment), list(materials))


# ---------------------------------------------------------------- 서버와 같은 입력


def test_system_and_user_are_built_by_server_code(rows):
    for row in rows[:200]:
        system, user, assistant = row["messages"]
        assert system == {"role": "system", "content": ADAPTER_SYSTEM_PROMPT}
        slots = mpt.slots_for(row["utterances"])
        context = WorkContext(
            text=full_text(slots), location=slots.location, hot_work_kind=slots.hot_work_kind,
            nearby_materials=slots.nearby_materials, headcount=slots.headcount, end_time=slots.end_time, equipment=slots.equipment,
        )
        assert user == {"role": "user", "content": build_adapter_input(context)}
        assert assistant["role"] == "assistant"


def test_user_message_shape(rows):
    for row in rows[:200]:
        user = row["messages"][1]["content"]
        assert user.startswith("작업 내용: " + row["utterances"][0])
        assert "\n".join(row["utterances"]) in user
        extra = [line for line in user.splitlines() if line.startswith(("주변 물질:", "투입 장비:", "장소:", "화기", "인원", "종료"))]
        assert all(line.startswith(("주변 물질:", "투입 장비:")) for line in extra), extra  # 짧은 입력은 이 두 줄만 덧붙인다


def test_assistant_is_the_three_key_json(rows):
    for row in rows:
        answer = json.loads(row["messages"][2]["content"])
        assert list(answer) == ["summary", "equipment", "materials"]
        assert isinstance(answer["summary"], str) and all(isinstance(x, str) for x in answer["equipment"] + answer["materials"])
        assert {k: answer[k] for k in answer} == {k: row["answer"][k] for k in answer}


def test_every_row_passes_server_validation(rows):
    for row in rows:
        text = "\n".join(row["utterances"])
        answer = json.loads(row["messages"][2]["content"])
        equipment, eq_dropped = pb.clean_equipment(answer["equipment"], text)
        materials, mat_dropped = pb.clean_materials(answer["materials"], text)
        assert eq_dropped == 0 and mat_dropped == 0, row
        assert len(equipment) == len(answer["equipment"]) and len(materials) == len(answer["materials"]), row
        assert pb.clean_summary(answer["summary"], text) == answer["summary"], row


def test_target_is_short_enough_for_server_token_limit(rows):
    # 서버는 생성을 BRIEF_MAX_TOKENS(160)에서 자른다. 한글은 글자당 0.4~0.6토큰(11/17 PC 측정)이라 150자면 넉넉히 안쪽이다.
    assert max(len(r["messages"][2]["content"]) for r in rows) <= 150
    assert pb.BRIEF_MAX_TOKENS >= 160


def test_summary_is_short_and_grounded(rows):
    from scripts.plan_eval.harness import summary_support

    for row in rows:
        text = "\n".join(row["utterances"])
        summary = row["answer"]["summary"]
        assert 4 <= len(summary) <= pb.BRIEF_SUMMARY_CHARS
        assert summary_support(summary, text)[0] == 1.0, row


# ---------------------------------------------------------------- 분포


def test_all_kinds_present_and_balanced(rows):
    counts = Counter(r["kind"] for r in rows)
    assert set(counts) == set(mpt.KIND_WEIGHTS)
    assert all(counts[k] >= 25 for k in counts), counts
    empty = sum(not r["answer"]["equipment"] and not r["answer"]["materials"] for r in rows) / len(rows)
    assert 0.10 <= empty <= 0.35  # 뽑을 것이 없는 문장도 충분히 있어야 지어내지 않는다
    multi = sum(len(r["utterances"]) > 1 for r in rows) / len(rows)
    assert 0.15 <= multi <= 0.45  # 되묻기 답까지 이은 여러 줄 입력


def test_negation_and_ood_coverage(rows):
    with_excluded = [r for r in rows if r["answer"]["excluded"]]
    assert len(with_excluded) >= 70
    for r in with_excluded:  # 쓰지 않는다고 한 것은 정답에 없다
        assert not set(r["answer"]["excluded"]) & set(r["answer"]["equipment"] + r["answer"]["materials"])
    ood = set(mpt.OOD_EQUIPMENT_TRAIN) | set(mpt.OOD_MATERIALS_TRAIN)
    assert sum(any(x in ood for x in r["answer"]["equipment"] + r["answer"]["materials"]) for r in rows) >= 100


def test_soft_negation_rows_keep_the_item_on_the_slot_line(rows):
    # '빼고 진행'·'쓸 일 없음'처럼 사전의 부정 규칙이 지금도 못 잡는 말: 슬롯 줄에는 들어가고 정답에서는 빠진다 (모델이 줄과 다르게 낼 수 있어야 한다)
    soft = [r for r in rows if r["answer"]["excluded"] and any(
        x in r["messages"][1]["content"].split("투입 장비:")[-1] for x in r["answer"]["excluded"])]
    assert len(soft) >= 10
    for r in soft:
        assert not set(r["answer"]["excluded"]) & set(json.loads(r["messages"][2]["content"])["equipment"])


def test_regex_caught_negation_never_reaches_the_slot_line(rows):
    for r in rows:
        line = r["messages"][1]["content"]
        for item in r["answer"]["excluded"]:
            if re.search(rf"{re.escape(item)}(은|는)?\s*(안 씀|사용 안 함|필요 없음|제외|쓸 일 없|쓸 계획 없|쓸 필요가 없|사용 금지)", " ".join(r["utterances"])):
                assert item not in line.split("작업 내용:")[1].split("\n")[-1] or "투입 장비" not in line


def test_h2s_and_server_dropped_names_are_not_gold(rows):
    for r in rows:
        assert "H2S" not in r["answer"]["materials"]  # 서버가 설비 번호 모양으로 보고 버리는 표기
    assert all("H2S" not in a for aliases in mpt.DICT_MATERIALS.values() for a in aliases)


# ---------------------------------------------------------------- 평가 문장과 겹침 막기


def _withdrawn(row):
    """앞 발화에서 부정 없이 말했다가 뒤 발화에서 거둔 항목들 (서버의 mentioned 로 판정)."""
    u = row["utterances"]
    found = []
    for item in row["answer"].get("excluded", []):
        used = [i for i, x in enumerate(u) if item in x and pb.mentioned(x, item)]
        dropped = [i for i, x in enumerate(u) if item in x and not pb.mentioned(x, item)]
        if used and dropped and min(used) < max(dropped):
            found.append(item)
    return found


def test_rows_that_withdraw_something_in_a_later_utterance_exist(rows):
    """실제 입력은 되묻기 답까지 이어 붙은 여러 발화이고, 뒤 발화가 앞의 말을 거둔다(T28). 학습에 이 모양이 있어야 한다."""
    withdrawn = [r for r in rows if _withdrawn(r)]
    assert len(withdrawn) >= 0.03 * len(rows), len(withdrawn)
    assert all(len(r["utterances"]) >= 2 for r in withdrawn)
    assert {r["kind"] for r in withdrawn} == {"negation"}
    # 사전 장비와 사전 밖 이름을 둘 다 거두는 줄이 있다
    items = {i for r in withdrawn for i in _withdrawn(r)}
    dictionary = {a for aliases in (*mpt.EQUIPMENT.values(), *mpt.MATERIALS.values()) for a in aliases}
    assert items & dictionary and items - dictionary


def test_withdrawn_items_are_not_gold_and_do_not_stay_on_the_equipment_line(rows):
    for row in rows:
        for item in _withdrawn(row):
            answer = row["answer"]
            assert item not in answer["equipment"] and item not in answer["materials"], row["utterances"]
            equipment_line = [x for x in row["messages"][1]["content"].splitlines() if x.startswith("투입 장비:")]
            if equipment_line:  # 슬롯이 장비는 뒤 발화의 거둠을 반영한다 (merge_slots)
                assert item not in equipment_line[0], row["utterances"]


def test_server_validation_cannot_tell_a_withdrawn_name_from_a_used_one(rows):
    """T28 2번의 기록: 거둔 이름도 서버의 mentioned 는 '언급됨'으로 본다. 그래서 정답에서 빼는 일은 모델이 해야 한다."""
    withdrawn = [(r, i) for r in rows for i in _withdrawn(r)]
    assert withdrawn
    for row, item in withdrawn:
        assert pb.mentioned("\n".join(row["utterances"]), item), row["utterances"]


def test_withdrawal_is_always_a_separate_later_utterance_even_without_random_split(guard, monkeypatch):
    monkeypatch.setattr(mpt, "MULTI_LINE_RATE", 0.0)
    made = mpt.generate(400, 5, guard)
    withdrawn = [r for r in made if _withdrawn(r)]
    assert withdrawn and all(len(r["utterances"]) >= 2 for r in withdrawn)
    assert sum(len(r["utterances"]) > 1 for r in made if not _withdrawn(r)) == 0  # 나눠 말하는 다른 줄은 확률 0이면 없다


def test_withdrawal_rows_are_checked_against_the_slot_line(guard):
    """거둔 장비가 슬롯 줄에 남는 줄(= 서버가 거두는 표현이 아닌 것)은 check_row 가 만들 때 멈춘다."""
    utter = ["도장 부스 점검. 그라인더로 연마하려고 했음.", "그라인더는 빼고 진행. 수작업으로 마감."]  # '빼고 진행'은 정규식이 지금도 못 잡는다 (T28 3번)
    slots = mpt.slots_for(utter)
    assert "그라인더" in slots.equipment
    sample = mpt.Sample("negation", [piece(text=utter[0]), piece("equip", utter[1], excluded=["그라인더"], later=True)], "도장 부스 점검")
    with pytest.raises(AssertionError):
        mpt.check_row(sample, utter, slots, "도장 부스 점검", [], [])


def test_no_eval_sentence_or_close_text(rows, valid_rows, guard):
    eval_texts = {" ".join(c.text.split()) for path in mpt.DEFAULT_EVAL_SETS for c in load_cases(path)}
    for r in [*rows, *valid_rows]:
        text = "\n".join(r["utterances"])
        assert " ".join(text.split()) not in eval_texts
        assert not guard.sentences & set(mpt._sentences(text)), text
        assert guard.max_similarity(text) < 0.6, text


def test_guard_rejects_copied_eval_text_and_sentence(guard):
    eval_text = load_cases(DEFAULT_EVAL_SET)[0].text
    assert guard.too_close(eval_text, 0.6)
    assert guard.too_close("다른 일 하고. " + eval_text.split(". ")[0] + ". 끝", 0.99)  # 문장 하나만 같아도 버린다
    assert not guard.too_close("완전히 다른 소화기 점검표 작성 진행.", 0.6)


def test_vocab_is_disjoint_from_eval_answers(guard):
    mpt.check_vocab(guard)
    train = set(mpt.OOD_EQUIPMENT_TRAIN) | set(mpt.OOD_MATERIALS_TRAIN)
    valid = set(mpt.OOD_EQUIPMENT_VALID) | set(mpt.OOD_MATERIALS_VALID)
    assert not train & valid and not (train | valid) & guard.ood_items


@pytest.mark.parametrize("name", ["염산", "핸드브레이커", "부탄올", "고압세척기"])
def test_check_vocab_rejects_names_from_eval(guard, monkeypatch, name):
    monkeypatch.setattr(mpt, "OOD_MATERIALS_TRAIN", [*mpt.OOD_MATERIALS_TRAIN, name])
    monkeypatch.setattr(mpt, "OOD_EQUIPMENT_TRAIN", [*mpt.OOD_EQUIPMENT_TRAIN, name])
    if name == "부탄올":  # 평가의 '부탄'을 품은 이름
        assert "부탄" in guard.ood_items
    with pytest.raises(AssertionError):
        mpt.check_vocab(guard)


@pytest.mark.parametrize("name", ["시클로헥산", "포크레인", "해머드릴", "염화수소", "디클로로메탄"])
def test_check_vocab_rejects_names_that_contain_dictionary_words(guard, monkeypatch, name):
    # 사전 단어를 품은 이름은 슬롯 줄에 엉뚱한 사전 항목을 넣어서 정답과 어긋난다 (T25로 못 막는 것들)
    monkeypatch.setattr(mpt, "OOD_MATERIALS_TRAIN", [*mpt.OOD_MATERIALS_TRAIN, name])
    monkeypatch.setattr(mpt, "OOD_EQUIPMENT_TRAIN", [*mpt.OOD_EQUIPMENT_TRAIN, name])
    with pytest.raises(AssertionError):
        mpt.check_vocab(guard)


def test_valid_uses_only_held_out_names(rows, valid_rows):
    train_text = " ".join("\n".join(r["utterances"]) for r in rows)
    valid_text = " ".join("\n".join(r["utterances"]) for r in valid_rows)
    for name in (*mpt.OOD_EQUIPMENT_VALID, *mpt.OOD_MATERIALS_VALID):
        assert name not in train_text, name
    for name in (*mpt.OOD_EQUIPMENT_TRAIN, *mpt.OOD_MATERIALS_TRAIN):
        assert name not in valid_text, name
    assert any(name in valid_text for name in (*mpt.OOD_EQUIPMENT_VALID, *mpt.OOD_MATERIALS_VALID))


def test_train_and_valid_do_not_share_texts(guard):
    seen: set[str] = set()
    a = mpt.generate(120, 1, guard, seen=seen)
    b = mpt.generate(60, 2, guard, holdout=True, seen=seen)
    assert not {"\n".join(r["utterances"]) for r in a} & {"\n".join(r["utterances"]) for r in b}


# ---------------------------------------------------------------- 재현성


def test_same_seed_same_rows_and_other_seed_differs(guard):
    a = mpt.generate(60, 5, guard)
    assert a == mpt.generate(60, 5, guard)
    assert a != mpt.generate(60, 6, guard)


# ---------------------------------------------------------------- 도우미


@pytest.mark.parametrize("word,pair,expected", [
    ("크레인", "으로/로", "크레인으로"), ("용접기", "으로/로", "용접기로"), ("그라인더", "으로/로", "그라인더로"), ("톨루엔", "으로/로", "톨루엔으로"),
    ("시너", "으로/로", "시너로"), ("아세톤", "은/는", "아세톤은"), ("시너", "은/는", "시너는"), ("프로판", "이/가", "프로판이"), ("LPG", "이/가", "LPG가"),
    ("벤젠", "으로/로", "벤젠으로"), ("염산", "으로/로", "염산으로"), ("과망간산칼륨", "으로/로", "과망간산칼륨으로"),
])
def test_josa(word, pair, expected):
    assert mpt.josa(word, pair) == expected


def test_ckey_and_used_treat_alias_surfaces_as_one_item():
    assert mpt.ckey("그라인다") == mpt.ckey("그라인더") == "그라인더"
    assert mpt.ckey("신나") == "시너" and mpt.ckey("염산") == "염산"
    assert mpt.used([piece("equip", "x", equipment=["그라인더"])], "그라인다")
    assert mpt.used([piece("equip", "x", excluded=["크레인"])], "크레인")
    assert not mpt.used([piece("equip", "x", equipment=["그라인더"])], "토치")


def test_join_adds_missing_periods_without_leading_space():
    assert mpt._join(["가 점검", "나 사용."]) == "가 점검. 나 사용."
    assert mpt._join(["가 점검.", "나 사용"]) == "가 점검. 나 사용"
    assert mpt._join(["하나"]) == "하나"


def test_t25_longest_name_is_used_for_slot_lines():
    slots = mpt.slots_for(["메탄올 드럼이 옆에 있음", "황화수소 확인"])
    assert slots.nearby_materials == ["메탄올", "황화수소"]  # '메탄'·'수소'가 따로 잡히지 않는다


def test_merge_slots_pipeline_is_used_for_multi_utterance_rows():
    slots = mpt.slots_for(["펌프실 밸브 교체", "토치 투입", "크레인 안 써요", "시너 있어요"])
    assert full_text(slots) == "펌프실 밸브 교체\n토치 투입\n크레인 안 써요\n시너 있어요"
    assert slots.equipment == ["토치"] and slots.nearby_materials == ["시너"]


# ---------------------------------------------------------------- 검사가 잘못된 정답을 거른다


def test_check_row_accepts_a_good_row():
    run_check(["펌프실 밸브 교체. 토치로 가열함."], summary="펌프실 밸브 교체", equipment=["토치"])


@pytest.mark.parametrize("kwargs,why", [
    (dict(summary="펌프실 밸브 교체", equipment=["용접기"]), "정답 항목이 원문에 없음"),
    (dict(summary="펌프실 밸브 교체", equipment=["토치"], materials=["톨루엔"]), "물질이 원문에 없음"),
    (dict(summary="펌프실 3개 밸브 교체", equipment=["토치"]), "요약에 원문에 없는 숫자"),
    (dict(summary="펌프실 배관 교체", equipment=["토치"]), "요약에 원문에 없는 말"),
    (dict(summary="밸브", equipment=["토치"]), "요약이 너무 짧음"),
    (dict(summary="펌프실 밸브 교체", equipment=[]), "슬롯 줄에는 있는 사전 장비가 정답에 없음"),
    (dict(summary="펌프실 밸브 교체", equipment=["토치", "토치"]), "표준 이름이 같은 항목 중복은 서버가 하나로 합친다"),
])
def test_check_row_rejects_bad_gold(kwargs, why):
    with pytest.raises(AssertionError):
        run_check(["펌프실 밸브 교체. 토치로 가열함."], **kwargs)


def test_check_row_rejects_negated_gold_item():
    with pytest.raises(AssertionError):
        run_check(["펌프실 밸브 교체. 토치는 안 써요."], summary="펌프실 밸브 교체", equipment=["토치"])


def test_check_row_rejects_non_material_gold():
    for bad in ("플랜지", "T-105", "톨루엔 탱크"):
        with pytest.raises(AssertionError):
            run_check([f"펌프실 {bad} 점검. 톨루엔 보관."], summary="펌프실 점검", materials=[bad])


def test_check_row_rejects_tag_like_h2s():
    with pytest.raises(AssertionError):  # 서버가 설비 번호 모양으로 보고 버리는 표기
        run_check(["펌프실 점검. H2S 냄새 확인."], summary="펌프실 점검", materials=["H2S"])


def test_check_row_rejects_ood_name_that_contains_a_dictionary_word():
    # '시클로헥산'을 정답으로 하면 슬롯 줄에 '헥산'이 들어가 어긋난다 → 이런 이름은 만들지 않는다
    with pytest.raises(AssertionError):
        run_check(["펌프실 점검. 시클로헥산 보관."], summary="펌프실 점검", materials=["시클로헥산"])


def test_check_row_requires_soft_marker_for_slot_only_items():
    soft = piece("equip", "토치는 빼고 진행함.", excluded=["토치"], soft=True)
    hard = piece("equip", "토치는 빼고 진행함.", excluded=["토치"])
    base = piece(text="펌프실 밸브 교체.")
    run_check(["펌프실 밸브 교체. 토치는 빼고 진행함."], summary="펌프실 밸브 교체", pieces=[base, soft])  # 슬롯 줄에 토치가 들어가는 걸 인정
    with pytest.raises(AssertionError):
        run_check(["펌프실 밸브 교체. 토치는 빼고 진행함."], summary="펌프실 밸브 교체", pieces=[base, hard])


def test_polite_forms_are_now_caught_so_they_are_not_soft_anymore():
    """11/17 에 규칙이 정중형을 잡게 됐다: 같은 표현을 soft(슬롯에 남음)로 표시하면 만들 때 멈춘다."""
    soft = piece("equip", "토치는 쓰지 않음.", excluded=["토치"], soft=True)
    base = piece(text="펌프실 밸브 교체.")
    with pytest.raises(AssertionError):
        run_check(["펌프실 밸브 교체. 토치는 쓰지 않음."], summary="펌프실 밸브 교체", pieces=[base, soft])


def test_prohibit_forms_are_now_caught_so_they_are_not_soft_anymore():
    """11/17(T30) 규칙이 '쓸 일 없'·'사용 금지'를 잡게 됐다: 같은 표현을 soft 로 표시하면 만들 때 멈춘다."""
    base = piece(text="펌프실 밸브 교체.")
    for phrase in ("토치 쓸 일 없음.", "토치 쓸 계획 없음.", "토치는 쓸 필요가 없음.", "토치 사용 금지."):
        soft = piece("equip", phrase, excluded=["토치"], soft=True)
        with pytest.raises(AssertionError):
            run_check([f"펌프실 밸브 교체. {phrase}"], summary="펌프실 밸브 교체", pieces=[base, soft])
        run_check([f"펌프실 밸브 교체. {phrase}"], summary="펌프실 밸브 교체", pieces=[base, piece("equip", phrase, excluded=["토치"])])


def test_every_soft_phrase_is_really_missed_by_the_rule_and_every_polite_one_is_caught(guard):
    """soft 표현은 규칙이 못 잡아야 하고(슬롯에 남음), 일반 부정 표현은 잡아야 한다 (규칙을 넓히면 이 목록을 다시 맞춘다)."""
    from app.agent.text_rules import EQUIPMENT, find_terms

    mk = mpt.Maker(7, holdout=False)
    for _ in range(60):
        soft_piece = mk.negate_equipment("드릴", soft=True)
        assert find_terms(soft_piece.text, EQUIPMENT)[0] == ["드릴"], soft_piece.text
        hard_piece = mk.negate_equipment("드릴")
        assert find_terms(hard_piece.text, EQUIPMENT)[0] == [], hard_piece.text
    polite = ["안 씁니다", "안 썼음", "쓰지 않음", "필요하지 않음", "투입하지 않음", "쓰지 않기로 함"]
    seen = {" ".join(mk.negate_equipment("드릴").text for _ in range(200)).count(w) > 0 for w in polite}
    assert seen == {True}  # 정중형이 데이터에 실제로 나온다


# ---------------------------------------------------------------- 실행 (CLI)


def test_cli_sample_prints_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "data"
    assert mpt.main(["--out", str(out), "--train", "40", "--valid", "10", "--sample", "1"]) == 0
    text = capsys.readouterr().out
    assert "[plain]" in text and "정답:" in text and "입력 줄:" in text
    assert not out.exists()


def test_cli_writes_files_with_counts_and_lf(tmp_path, capsys):
    out = tmp_path / "data"
    assert mpt.main(["--out", str(out), "--train", "50", "--valid", "12"]) == 0
    train = (out / "train.jsonl").read_bytes()
    valid = (out / "valid.jsonl").read_bytes()
    assert b"\r" not in train and train.endswith(b"\n")
    assert len(train.decode("utf-8").splitlines()) == 50 and len(valid.decode("utf-8").splitlines()) == 12
    row = json.loads(train.decode("utf-8").splitlines()[0])
    assert set(row) == {"messages", "kind", "utterances", "answer"}
    assert "평가 문장과 가장 비슷한" in capsys.readouterr().out


def test_cli_is_reproducible_byte_for_byte(tmp_path):
    for name in ("a", "b"):
        assert mpt.main(["--out", str(tmp_path / name), "--train", "40", "--valid", "8"]) == 0
    for f in ("train.jsonl", "valid.jsonl"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_generate_refuses_when_the_polite_negation_rule_is_missing_or_off(guard, monkeypatch):
    """규칙 없이 만들면 정중형 정답이 슬롯 줄과 어긋난다: 이유를 알기 어려운 AssertionError 대신 처음에 알려 준다."""
    monkeypatch.setattr(mpt.text_rules, "POLITE_NEGATION", False)
    with pytest.raises(RuntimeError, match="apply_polite_negation"):
        mpt.generate(5, 1, guard)
    monkeypatch.delattr(mpt.text_rules, "POLITE_NEGATION")  # 규칙 코드가 아예 없는 옛 app
    with pytest.raises(RuntimeError, match="POLITE_NEGATION"):
        mpt.generate(5, 1, guard)


def test_generate_fails_loudly_when_it_cannot_make_enough_distinct_rows(guard, monkeypatch):
    monkeypatch.setattr(mpt, "NONE_TASKS", ["소화기 점검표 작성"])
    monkeypatch.setattr(mpt, "KIND_WEIGHTS", {"none": 1})
    monkeypatch.setattr(mpt, "LOCATIONS", ["펌프실"])
    monkeypatch.setattr(mpt, "HEADS", ["오늘"])
    monkeypatch.setattr(mpt, "EXTRAS", ["2명 작업"])
    monkeypatch.setattr(mpt, "ENDINGS", [("함", "")])
    with pytest.raises(RuntimeError, match="재료가 모자랍니다"):
        mpt.generate(60, 1, guard)  # 만들 수 있는 문장이 이보다 훨씬 적게 되도록 재료를 줄여 놓았다 (시도 한도 = 건수 × 60)


# ---------------------------------------------------------------- 11/17 (T30): 규칙 v2 · 어려운 평가 문장 · 말 많은 문장


def test_default_eval_guard_covers_all_three_sets_and_79_sentences(guard):
    assert [p.name for p in mpt.DEFAULT_EVAL_SETS] == ["eval_set.jsonl", "eval_set_multiline.jsonl", "eval_set_user_hard.jsonl"]
    assert len(guard.texts) == 79
    assert any("굴뚝 청소 작업" in t for t in guard.texts)  # 실제 현장 문장(u02)도 누설 방지 대상이다


def test_no_generated_text_is_close_to_the_user_hard_sentences(rows, valid_rows):
    hard = mpt.EvalGuard.load(mpt.USER_HARD_EVAL_SET)
    for r in [*rows, *valid_rows]:
        text = "\n".join(r["utterances"])
        assert not hard.too_close(text, 0.6), text


@pytest.mark.parametrize("name", ["윈치", "헬륨", "스프레이건", "고압세척기", "질산", "초산", "아세트산", "황산", "염산"])
def test_check_vocab_rejects_hard_set_answers_and_names_now_in_the_dictionary(guard, monkeypatch, name):
    """어려운 평가 문장의 정답(윈치·헬륨 …)과 사전에 들어온 약품(질산·초산 …)은 '사전 밖' 목록에 둘 수 없다."""
    monkeypatch.setattr(mpt, "OOD_MATERIALS_VALID", [*mpt.OOD_MATERIALS_VALID, name])
    monkeypatch.setattr(mpt, "OOD_EQUIPMENT_VALID", [*mpt.OOD_EQUIPMENT_VALID, name])
    with pytest.raises(AssertionError):
        mpt.check_vocab(guard)


def test_generate_refuses_when_the_v2_rules_are_missing_or_off(guard, monkeypatch):
    for name in ("EXTENDED_VOCAB", "PROHIBIT_NEGATION"):
        with monkeypatch.context() as m:
            m.setattr(mpt.text_rules, name, False)
            with pytest.raises(RuntimeError, match=name):
                mpt.generate(5, 1, guard)
        with monkeypatch.context() as m:
            m.delattr(mpt.text_rules, name)  # 규칙 코드가 아예 없는 옛 app
            with pytest.raises(RuntimeError, match="apply_rules_v2"):
                mpt.generate(5, 1, guard)


def test_kind_list_and_weights_agree_and_verbose_is_a_train_only_kind():
    assert set(mpt._KINDS) == set(mpt.KIND_WEIGHTS)  # noqa: SLF001
    from scripts.plan_eval import harness

    assert "verbose" not in harness.KINDS  # 평가 유형이 아니다 (평가 표의 유형별 점수에 새 칸이 생기지 않게)


def test_new_dictionary_words_are_used_in_training_rows(rows):
    text = "\n".join("\n".join(r["utterances"]) for r in rows)
    for name in ("사다리차", "황산", "염산", "질산", "초산"):
        assert text.count(name) >= 3, name


def test_ladder_truck_rows_put_the_truck_on_the_equipment_line_not_the_ladder(rows):
    seen = 0
    for r in rows:
        if "사다리차" in "\n".join(r["utterances"]):
            line = r["messages"][1]["content"].split("투입 장비:")[-1].split("\n")[0] if "투입 장비:" in r["messages"][1]["content"] else ""
            assert "사다리차" in line or "사다리차" in r["answer"]["excluded"], r["utterances"]
            assert "사다리," not in line + "," or "사다리차" in line
            seen += 1
    assert seen >= 3


def test_prohibit_and_no_need_forms_appear_and_never_reach_the_slot_line(rows):
    phrases = ("사용 금지", "쓸 일 없", "쓸 계획 없", "쓸 필요가 없")
    hits = [r for r in rows if any(p in "\n".join(r["utterances"]) for p in phrases)]
    assert len(hits) >= 10, len(hits)
    for r in hits:
        slot_line = r["messages"][1]["content"].split("투입 장비:")[-1] if "투입 장비:" in r["messages"][1]["content"] else ""
        for item in r["answer"]["excluded"]:
            if item in mpt.EQUIPMENT:
                assert item not in slot_line, r["utterances"]


def test_verbose_rows_are_wordy_but_the_summary_is_only_the_main_task(rows):
    verbose = [r for r in rows if r["kind"] == "verbose"]
    assert len(verbose) >= 60
    past = re.compile(r"(했음|했습니다|했어요|하였음|마침|끝냄|완료함)\.")
    assert sum(bool(past.search(r["utterances"][0])) for r in verbose) / len(verbose) >= 0.9  # 작업 문장이 과거형 서술이다
    for r in verbose:
        summary = json.loads(r["messages"][2]["content"])["summary"]
        assert not re.search(r"(했|였|마침|끝냄|완료)", summary), summary  # 끝맺음·동사를 요약에 끌고 오지 않는다 (잘린 동사 방지)
        assert len(r["utterances"][0]) > len(summary)
    plain = [r for r in rows if r["kind"] == "plain"]
    mean = lambda rs: sum(len("\n".join(r["utterances"])) for r in rs) / len(rs)  # noqa: E731
    assert mean(verbose) > mean(plain)  # 평균적으로 말이 더 많다
    for r in verbose:  # 인원 이야기는 한 번만 (두 번 나오면 숫자가 서로 어긋난다)
        assert len(set(re.findall(r"(\d+)명", "\n".join(r["utterances"])))) <= 1, r["utterances"]
    assert any(json.loads(r["messages"][2]["content"])["equipment"] for r in verbose)
    assert any(json.loads(r["messages"][2]["content"])["materials"] for r in verbose)
    assert any(re.search(r", \d+명$", json.loads(r["messages"][2]["content"])["summary"]) for r in verbose)  # 인원이 든 요약 틀도 있다


def test_every_new_negation_form_is_actually_produced():
    mk = mpt.Maker(11)
    made = " ".join(mk.negate_equipment("드릴").text for _ in range(600))
    for phrase in ("사용 금지.", "사용 금지 상태라", "쓸 일 없음", "쓸 계획 없음", "쓸 필요가 없음", "쓸 일 없어서", "안 씁니다", "쓰지 않기로 함"):
        assert phrase in made, phrase
    soft = " ".join(mk.negate_equipment("드릴", soft=True).text for _ in range(300))
    for phrase in ("빼고 진행함", "가져오지 않음", "안 가져옴", "반입하지 않음", "반입 안 함"):
        assert phrase in soft, phrase


def test_verbose_fillers_contain_no_dictionary_words_and_no_negation_after_a_word():
    from app.agent.text_rules import EQUIPMENT, MATERIALS, find_terms

    for text in [*mpt.VERBOSE_FILLERS, *mpt.VERBOSE_REASONS]:
        assert find_terms(text, EQUIPMENT) == ([], []) and find_terms(text, MATERIALS) == ([], []), text
    mk = mpt.Maker(5)
    for _ in range(300):
        for piece_ in (mk.verbose_equipment("그라인더", "그라인더"), mk.verbose_material("톨루엔")):
            found = find_terms(piece_.text, {**mpt.EQUIPMENT, **mpt.MATERIALS})
            assert found[1] == [], piece_.text  # 쓴다고 쓴 말이 부정으로 읽히면 정답과 슬롯 줄이 어긋난다


def test_source_hygiene():
    source = Path(mpt.__file__).read_text(encoding="utf-8")
    assert "\r" not in source
    assert "import torch" not in source and "transformers" not in source  # 생성기는 모델 없이 돈다
