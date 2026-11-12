"""문장 추출 어댑터 학습 데이터 생성기 (scripts/make_extract_train.py). 모델·네트워크 없이 돈다."""

from __future__ import annotations

from app.agent.rule_input_parse import format_lines, lines_to_raw, parse_rule_input
from app.agent.rule_input_prompt import ADAPTER_SYSTEM_PROMPT
from scripts import make_extract_train as mt

ITEMS = list(mt.ITEMS)
TYPES = ["HOT", "CONFINED", "HEIGHT"]


def _rows(n=400, seed=7):
    texts, names = mt.load_gold([mt.GOLD_V2])
    return mt.generate(n, seed, texts, names), texts, names


def test_same_seed_same_data():
    a, _, _ = _rows(200, 3)
    b, _, _ = _rows(200, 3)
    assert a == b


def test_rows_match_runtime_prompt_and_parse_cleanly():
    rows, _, _ = _rows()
    for row in rows:
        system, user, assistant = row["messages"]
        assert system == {"role": "system", "content": ADAPTER_SYSTEM_PROMPT}   # 실행과 같은 문구
        assert user["role"] == "user" and user["content"].startswith("문장: ")
        text = user["content"].removeprefix("문장: ")
        out = row["answer"]
        assert assistant["content"] == format_lines(out)                         # 11/12: 줄 형식
        parsed = parse_rule_input(lines_to_raw(assistant["content"]), text, ITEMS, TYPES)   # ai-model 파서가 하나도 버리지 않는다
        assert parsed.dropped == [] and len(parsed.measurements) == len(
            {(m["item"], m["value"]) for m in out["measurements"]}), (text, parsed.dropped)
        assert len(parsed.work_types) == len(out["workTypes"])
        assert parsed.watcher_name == out["watcherName"]


def test_no_gold_sentences_or_names():
    rows, gold_texts, gold_names = _rows(800)
    gold = set(gold_texts)
    for row in rows:
        text = row["messages"][1]["content"].removeprefix("문장: ")
        assert text not in gold
        name = row["answer"]["watcherName"]
        assert name not in gold_names


def test_every_tag_and_rule_appears():
    rows, _, _ = _rows(800)
    assert {r["tag"] for r in rows} == set(mt.TAG_WEIGHTS)
    outs = [r["answer"] for r in rows]
    assert any(m["unit"] is None for o in outs for m in o["measurements"])                 # 단위 없음 → null
    assert any(len({m["item"] for m in o["measurements"]}) < len(o["measurements"]) for o in outs)   # 값 두 번 → 둘 다
    assert any(o["measurements"] == [] and o["workTypes"] == [] for o in outs)            # 뽑을 게 없는 문장
    assert any(r["messages"][2]["content"] == "없음" for r in rows)                         # → "없음" 한 줄


def test_new_cases_from_gold_v2_misses_appear():
    """11/12 보강: 한 덩어리 두 종류, 밀폐 아닌 탱크 주변, 숫자 아닌 값 + 실제 값, 괄호 어순"""
    rows, _, _ = _rows(1500)
    outs = [(r["messages"][1]["content"], r["answer"]) for r in rows]
    kinds = [{w["type"] for w in o["workTypes"]} for _, o in outs]
    assert any(k == {"HOT", "CONFINED"} for k in kinds) and any(k == {"HOT", "HEIGHT"} for k in kinds)
    assert any(any(w in t for w in ("외벽", "주변", "바깥", "외부")) and not o["workTypes"] for t, o in outs)
    assert any("정상" in t and o["measurements"] for t, o in outs)
    assert any("(" in m["evidence"] for _, o in outs for m in o["measurements"])


def test_cli_writes_train_and_valid(tmp_path):
    assert mt.main(["--out", str(tmp_path), "--train", "30", "--valid", "5"]) == 0
    assert len((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 30
    assert len((tmp_path / "valid.jsonl").read_text(encoding="utf-8").splitlines()) == 5
