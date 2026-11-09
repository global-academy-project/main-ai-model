"""scripts/eval_retrieval.py 시험. 네트워크·DB 없이 가짜 검색 함수로 돌린다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import eval_retrieval as ev  # noqa: E402

CASES_FILE = ROOT / "eval" / "retrieval_law_v0.jsonl"
CHUNKS_FILE = ROOT / "out" / "law" / "05_chunks.jsonl"


def hit(article: str, score: float = 0.6) -> dict:
    return {"id": "x", "doc_id": "LAW:1", "source": f"산업안전보건기준에 관한 규칙 ({article})", "text": f"{article.split()[0]}(제목) 본문", "score": score}


def case(**over) -> dict:
    base = {"id": "T1", "kind": "human", "category": "화기", "query": "q", "expect_articles": ["제241조의2"], "confidence": "high"}
    base.update(over)
    return base


def fake_search(mapping):
    def search(query, *, k, filters):
        return mapping[query][:k]

    return search


# ── 조 번호 읽기 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hit_, expected",
    [
        ({"source": "산업안전보건기준에 관한 규칙 (제241조의2)", "text": ""}, "제241조의2"),
        ({"source": "산업안전보건기준에 관한 규칙 (제42조 제1항~제3항)", "text": ""}, "제42조"),
        ({"source": None, "text": "제13조(안전난간의 구조) 사업주는"}, "제13조"),
        ({"source": "제목만", "text": "본문"}, None),
    ],
)
def test_article_of(hit_, expected):
    assert ev.article_of(hit_) == expected


def test_article_exact_match_not_prefix():
    """제241조 기대에 제241조의2가 맞은 것으로 세지면 안 된다."""
    res = ev.evaluate_case(case(expect_articles=["제241조"]), fake_search({"q": [hit("제241조의2")]}), k_fetch=5, filters={})
    assert res["rank_chunk"] is None


# ── 순위·지표 ───────────────────────────────────────────────────────────────────


def test_ranks_chunk_vs_article():
    hits = [hit("제10조"), hit("제10조 제1항"), hit("제241조의2")]
    res = ev.evaluate_case(case(), fake_search({"q": hits}), k_fetch=5, filters={})
    assert res["rank_chunk"] == 3  # 청크 기준은 3번째
    assert res["rank_article"] == 2  # 같은 조의 청크는 하나로 묶어 2번째


def test_any_of_expected_counts():
    res = ev.evaluate_case(case(expect_articles=["제241조", "제236조"]), fake_search({"q": [hit("제236조")]}), k_fetch=5, filters={})
    assert res["rank_chunk"] == 1


def test_not_found():
    res = ev.evaluate_case(case(), fake_search({"q": [hit("제1조")]}), k_fetch=5, filters={})
    assert res["rank_chunk"] is None and res["rank_article"] is None


def test_summary_values():
    results = [
        {"rank_chunk": 1, "rank_article": 1, "error": None},
        {"rank_chunk": 4, "rank_article": 2, "error": None},
        {"rank_chunk": None, "rank_article": None, "error": None},
        {"rank_chunk": 7, "rank_article": 5, "error": None},
    ]
    s = ev.summarize(results)
    assert s["n"] == 4
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.25 and s["hit@5"] == 0.5 and s["hit@10"] == 0.75
    assert s["mrr"] == round((1 + 1 / 2 + 0 + 1 / 5) / 4, 4)


def test_summary_excludes_errors():
    results = [{"rank_chunk": 1, "rank_article": 1, "error": None}, {"rank_chunk": None, "rank_article": None, "error": "X"}]
    assert ev.summarize(results)["n"] == 1


def test_k_fetch_limits_search():
    seen = {}

    def search(query, *, k, filters):
        seen["k"], seen["filters"] = k, filters
        return []

    ev.evaluate_case(case(), search, k_fetch=7, filters={"collection": "kb_guides"})
    assert seen == {"k": 7, "filters": {"collection": "kb_guides"}}


def test_search_error_marked_not_raised():
    def boom(query, *, k, filters):
        raise RuntimeError("검색 실패")

    results = ev.run([case()], boom, k_fetch=5, filters={})
    assert results[0]["error"].startswith("RuntimeError")
    assert ev.summarize(results) == {"n": 0}


def test_group_summary():
    results = ev.run(
        [case(id="A", category="화기"), case(id="B", category="고소", expect_articles=["제44조"])],
        fake_search({"q": [hit("제241조의2")]}),
        k_fetch=5,
        filters={},
    )
    groups = ev.group_summary(results, "category")
    assert groups["화기"]["hit@1"] == 1.0 and groups["고소"]["hit@10"] == 0.0


def test_compare_better_and_worse():
    cur = [
        {"id": "A", "rank_chunk": 1, "error": None},
        {"id": "B", "rank_chunk": None, "error": None},
        {"id": "C", "rank_chunk": 3, "error": None},
        {"id": "D", "rank_chunk": 2, "error": None},
    ]
    prev = [
        {"id": "A", "rank_chunk": 4, "error": None},
        {"id": "B", "rank_chunk": 2, "error": None},
        {"id": "C", "rank_chunk": 3, "error": None},
    ]
    diff = ev.compare(cur, prev)
    assert diff["better"] == ["A: 4위 → 1위"]
    assert diff["worse"] == ["B: 2위 → 못 찾음"]  # D는 이전에 없어 제외, C는 같음


# ── 정답 세트 읽기 ──────────────────────────────────────────────────────────────


def write_cases(tmp_path, rows):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_cases_ok(tmp_path):
    assert len(ev.load_cases(write_cases(tmp_path, [case(), case(id="T2")]))) == 2


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"kind": "other"}, "kind"),
        ({"confidence": "none"}, "confidence"),
        ({"expect_articles": []}, "expect_articles"),
        ({"expect_articles": ["241조"]}, "형식"),
        ({"expect_articles": ["제241조 제2항"]}, "형식"),
        ({"query": " "}, "query"),
    ],
)
def test_load_cases_rejects(tmp_path, over, needle):
    with pytest.raises(ValueError, match=needle):
        ev.load_cases(write_cases(tmp_path, [case(**over)]))


def test_load_cases_duplicate_id(tmp_path):
    with pytest.raises(ValueError, match="중복"):
        ev.load_cases(write_cases(tmp_path, [case(), case()]))


def test_load_cases_bad_json(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        ev.load_cases(p)


def test_missing_articles():
    assert ev.missing_articles([case(expect_articles=["제1조", "제999조"])], {"제1조"}) == [("T1", "제999조")]


def test_articles_in_chunks(tmp_path):
    p = tmp_path / "k.jsonl"
    rows = [{"locator": "제241조의2"}, {"locator": "제42조 제1항~제3항"}, {"locator": ""}]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert ev.articles_in_chunks(p) == {"제241조의2", "제42조"}


# ── 실제 정답 세트 파일 ─────────────────────────────────────────────────────────


def test_shipped_cases_load_and_cover_kinds():
    cases = ev.load_cases(CASES_FILE)
    assert {c["kind"] for c in cases} == set(ev.KINDS)
    assert len(cases) >= 40


@pytest.mark.skipif(not CHUNKS_FILE.exists(), reason="out/law/05_chunks.jsonl 없음")
def test_shipped_cases_expected_articles_exist():
    assert ev.missing_articles(ev.load_cases(CASES_FILE), ev.articles_in_chunks(CHUNKS_FILE)) == []


def test_validate_mode_runs_without_network(capsys):
    assert ev.main(["--validate"]) == 0
    assert "형식 정상" in capsys.readouterr().out
