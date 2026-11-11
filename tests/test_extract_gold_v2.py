"""LLM 추출 평가 정답(eval/rule_input_extract_v2.json) 형식 검사와 채점 기준. 모델 없이 돈다."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from app.schemas.rule_input_extract import DEFAULT_ITEMS, DEFAULT_WORK_TYPES
from scripts.eval_rule_input_extract import _wanted

GOLD = Path(__file__).resolve().parents[1] / "eval" / "rule_input_extract_v2.json"
CASES = json.loads(GOLD.read_text(encoding="utf-8"))


def test_gold_shape():
    ids = [c["id"] for c in CASES]
    assert ids == list(range(1, len(CASES) + 1))
    assert len({c["text"] for c in CASES}) == len(CASES)            # 같은 문장 없음
    for c in CASES:
        assert c["tag"] and c["text"]
        assert set(c["workTypes"]) <= set(DEFAULT_WORK_TYPES)
        values = dict(c.get("measurement") or {})
        for item, vs in (c.get("duplicates") or {}).items():
            assert len(vs) >= 2 and item not in values, c["id"]     # 두 번인 항목은 duplicates 에만
        for item in list(values) + list(c.get("duplicates") or {}):
            assert item in DEFAULT_ITEMS, c["id"]
        assert "notes" not in c, c["id"]                            # v2 는 notes(비워야 맞음)를 쓰지 않는다
        watcher = (c.get("form") or {}).get("watcher_name")
        assert watcher is None or watcher in c["text"], c["id"]


def test_values_are_written_in_text():
    for c in CASES:
        nums = [v for v in (c.get("measurement") or {}).values()]
        nums += [v for vs in (c.get("duplicates") or {}).values() for v in vs]
        for v in nums:
            assert str(v) in c["text"], (c["id"], v)   # 적힌 숫자 그대로 (계산·환산 없음)


def test_wanted_counts_duplicates_and_v1_notes():
    assert _wanted({"measurement": {"H2S": 1}, "duplicates": {"O2": [18, 20.5]}}) == Counter(
        {("H2S", Decimal("1")): 1, ("O2", Decimal("18")): 1, ("O2", Decimal("20.5")): 1})
    # gold-v1: notes 항목은 비워야 맞음 (옛 기준)
    assert _wanted({"measurement": {"O2": 18}, "notes": ["O2"]}) == Counter()
    assert _wanted({"measurement": {"O2": 20.0}}) == Counter({("O2", Decimal("20")): 1})   # 20.0 과 20 은 같은 값
