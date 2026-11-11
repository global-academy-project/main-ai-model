"""검색 정답 세트 평가: 질문마다 정답 조문이 검색 상위 몇 번째에 나오는지 재고 hit@k·MRR을 낸다.

    python -m scripts.eval_retrieval --validate          # 네트워크 없이 정답 세트 형식과 기대 조문 존재만 검사
    python -m scripts.eval_retrieval                     # 실제 검색(임베딩 API + Qdrant 읽기 전용)으로 평가
    python -m scripts.eval_retrieval --out out/eval/baseline.json
    python -m scripts.eval_retrieval --compare out/eval/baseline.json --out out/eval/exp1.json

프로젝트 루트에서 실행한다. 검색 설정은 scripts/try_rag_tools.py와 같은 환경변수를 쓴다
(EMBEDDING_URL, EMBEDDING_API_KEY, QDRANT_URL, QDRANT_READ_ONLY_API_KEY 등). 읽기만 하며 아무것도 쓰지 않는다.

판정:
- 검색 결과는 청크 단위다. 조 번호는 hit의 source("제목 (제42조 제1항~제3항)")에서 읽는다.
- hit@k  : 상위 k개 청크 안에 기대 조문(expect_articles 중 하나)이 있는 질문의 비율 (에이전트가 받는 결과와 같은 기준)
- MRR    : 같은 조의 청크를 하나로 묶은 순위의 역수 평균 (못 찾으면 0)
- 정답 세트의 status가 candidate인 동안은 기대 조문이 전문가 확인 전의 후보다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "eval" / "retrieval_law_v0.jsonl"
DEFAULT_CHUNKS = ROOT / "out" / "law" / "05_chunks.jsonl"

ARTICLE_RE = re.compile(r"제\d+조(?:의\d+)?")
_ARTICLE_FULL_RE = re.compile(r"^제\d+조(?:의\d+)?$")
_SOURCE_RE = re.compile(r"\((제\d+조(?:의\d+)?)")
KINDS = ("human", "colloquial", "agent")
CONFIDENCES = ("high", "medium", "low")
HIT_KS = (1, 3, 5, 10)

SearchFn = Callable[..., list[dict[str, Any]]]


# ── 정답 세트 읽기·검사 ──────────────────────────────────────────────────────────


def load_cases(path: Path) -> list[dict[str, Any]]:
    """JSONL을 읽고 형식을 검사한다. 문제가 있으면 모아서 한 번에 ValueError로 알린다."""
    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{no}행: JSON이 아닙니다 ({exc.msg})")
                continue
            problems = _case_problems(case)
            if not problems and case["id"] in seen:
                problems.append(f"id 중복 {case['id']}")
            if problems:
                errors.append(f"{no}행: " + "; ".join(problems))
                continue
            seen.add(case["id"])
            cases.append(case)
    if errors:
        raise ValueError("정답 세트 형식 오류:\n  " + "\n  ".join(errors))
    if not cases:
        raise ValueError("정답 세트가 비어 있습니다.")
    return cases


def _case_problems(case: Any) -> list[str]:
    if not isinstance(case, dict):
        return ["객체가 아닙니다"]
    problems: list[str] = []
    for key in ("id", "category", "query"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            problems.append(f"{key}가 비었습니다")
    if case.get("kind") not in KINDS:
        problems.append(f"kind는 {KINDS} 중 하나여야 합니다")
    if case.get("confidence") not in CONFIDENCES:
        problems.append(f"confidence는 {CONFIDENCES} 중 하나여야 합니다")
    expect = case.get("expect_articles")
    if not isinstance(expect, list) or not expect:
        problems.append("expect_articles는 비어 있지 않은 목록이어야 합니다")
    else:
        for article in expect:
            if not isinstance(article, str) or not _ARTICLE_FULL_RE.match(article):
                problems.append(f"expect_articles 값 형식 오류: {article!r} (예: 제241조의2)")
    return problems


def articles_in_chunks(chunks_path: Path) -> set[str]:
    """청크 파일(05_chunks.jsonl)에 있는 조 번호 집합."""
    found: set[str] = set()
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            locator = json.loads(line).get("locator") or ""
            match = ARTICLE_RE.match(locator)
            if match:
                found.add(match.group(0))
    return found


def missing_articles(cases: Sequence[Mapping[str, Any]], known: set[str]) -> list[tuple[str, str]]:
    return [(c["id"], a) for c in cases for a in c["expect_articles"] if a not in known]


# ── 평가 ─────────────────────────────────────────────────────────────────────────


def article_of(hit: Mapping[str, Any]) -> str | None:
    """hit에서 조 번호를 읽는다: source의 '(제N조…' 우선, 없으면 본문 맨 앞."""
    source = hit.get("source")
    if isinstance(source, str):
        match = _SOURCE_RE.search(source)
        if match:
            return match.group(1)
    text = hit.get("text")
    if isinstance(text, str):
        match = ARTICLE_RE.match(text.lstrip())
        if match:
            return match.group(0)
    return None


def evaluate_case(case: Mapping[str, Any], search: SearchFn, *, k_fetch: int, filters: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    hits = search(case["query"], k=k_fetch, filters=dict(filters))
    ms = int((time.perf_counter() - started) * 1000)
    expected = set(case["expect_articles"])
    articles = [article_of(h) for h in hits]

    rank_chunk = next((i for i, a in enumerate(articles, 1) if a in expected), None)

    seen: list[str | None] = []
    for a in articles:  # 같은 조의 청크는 한 번만 센다
        if a not in seen:
            seen.append(a)
    rank_article = next((i for i, a in enumerate(seen, 1) if a in expected), None)

    return {
        "id": case["id"],
        "kind": case["kind"],
        "category": case["category"],
        "confidence": case["confidence"],
        "query": case["query"],
        "expect": list(case["expect_articles"]),
        "rank_chunk": rank_chunk,
        "rank_article": rank_article,
        "top": [{"article": a, "score": round(float(h.get("score", 0.0)), 4)} for a, h in zip(articles[:5], hits[:5])],
        "ms": ms,
        "error": None,
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r.get("error")]
    n = len(ok)
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    for k in HIT_KS:
        out[f"hit@{k}"] = round(sum(1 for r in ok if r["rank_chunk"] is not None and r["rank_chunk"] <= k) / n, 4)
    out["mrr"] = round(sum(1 / r["rank_article"] for r in ok if r["rank_article"]) / n, 4)
    return out


def group_summary(results: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    return {name: summarize(items) for name, items in groups.items()}


def run(
    cases: Sequence[Mapping[str, Any]],
    search: SearchFn,
    *,
    k_fetch: int,
    filters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(evaluate_case(case, search, k_fetch=k_fetch, filters=filters))
        except Exception as exc:  # 검색 실패는 해당 질문만 오류로 표시하고 계속한다 (메시지는 내부 예외 문구라 키가 들어가지 않는다)
            results.append(
                {
                    "id": case["id"], "kind": case["kind"], "category": case["category"], "confidence": case["confidence"],
                    "query": case["query"], "expect": list(case["expect_articles"]),
                    "rank_chunk": None, "rank_article": None, "top": [], "ms": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def compare(current: Sequence[Mapping[str, Any]], previous: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """이전 결과와 비교해 순위가 좋아지거나 나빠진 질문을 돌려준다 (못 찾음은 무한대로 본다)."""
    prev = {r["id"]: r for r in previous}
    inf = 10**9
    better: list[str] = []
    worse: list[str] = []
    for r in current:
        p = prev.get(r["id"])
        if p is None or r.get("error") or p.get("error"):
            continue
        now = r["rank_chunk"] or inf
        before = p["rank_chunk"] or inf
        label = f"{r['id']}: {_fmt_rank(p['rank_chunk'])} → {_fmt_rank(r['rank_chunk'])}"
        if now < before:
            better.append(label)
        elif now > before:
            worse.append(label)
    return {"better": better, "worse": worse}


def _fmt_rank(rank: int | None) -> str:
    return f"{rank}위" if rank else "못 찾음"


# ── 출력 ─────────────────────────────────────────────────────────────────────────


def _print_results(results: Sequence[Mapping[str, Any]]) -> None:
    for r in results:
        if r.get("error"):
            print(f"! {r['id']:<5} 오류  {r['query']}\n      {r['error']}")
            continue
        mark = "O" if r["rank_chunk"] and r["rank_chunk"] <= 5 else "X"
        top = " ".join(f"{t['article']}({t['score']:.2f})" for t in r["top"][:3])
        print(
            f"{mark} {r['id']:<5} 청크 {_fmt_rank(r['rank_chunk']):<6} 조 {_fmt_rank(r['rank_article']):<6} "
            f"기대 {','.join(r['expect'])} [{r['confidence']}]\n      {r['query']}\n      상위: {top}"
        )


def _print_summary(title: str, summary: Mapping[str, Any]) -> None:
    if not summary.get("n"):
        print(f"{title}: 결과 없음")
        return
    hits = "  ".join(f"hit@{k} {summary[f'hit@{k}']:.2f}" for k in HIT_KS)
    print(f"{title:<14} n={summary['n']:<3} {hits}  MRR {summary['mrr']:.3f}")


def _filter_cases(cases: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    out = list(cases)
    if args.kind:
        out = [c for c in out if c["kind"] in args.kind]
    if args.category:
        out = [c for c in out if c["category"] in args.category]
    if args.confidence:
        out = [c for c in out if c["confidence"] in args.confidence]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="기대 조문 존재 검사용 05_chunks.jsonl")
    parser.add_argument("--validate", action="store_true", help="네트워크 없이 형식과 기대 조문 존재만 검사")
    parser.add_argument("--k-fetch", type=int, default=20, help="질문마다 가져올 청크 수 (hit@10까지 재려면 10 이상)")
    parser.add_argument("--collection", default="kb_guides")
    parser.add_argument("--source-type", default="LAW")
    parser.add_argument("--kind", action="append", choices=KINDS, help="이 종류만 (여러 번 가능)")
    parser.add_argument("--category", action="append", help="이 분야만 (여러 번 가능)")
    parser.add_argument("--confidence", action="append", choices=CONFIDENCES, help="기대 조문 확신도로 거르기")
    parser.add_argument("--out", type=Path, help="결과를 JSON으로 저장")
    parser.add_argument("--compare", type=Path, help="이전 결과 JSON과 순위 비교")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        print(exc)
        return 2

    if args.chunks.exists():
        missing = missing_articles(cases, articles_in_chunks(args.chunks))
        if missing:
            print("청크 파일에 없는 기대 조문:", ", ".join(f"{i}→{a}" for i, a in missing))
            return 2
        print(f"정답 세트 {len(cases)}개 형식 정상, 기대 조문이 모두 청크에 있음 ({args.chunks.name})")
    else:
        print(f"정답 세트 {len(cases)}개 형식 정상 (청크 파일 {args.chunks}이(가) 없어 기대 조문 존재 검사는 건너뜀)")
    if args.validate:
        return 0

    selected = _filter_cases(cases, args)
    if not selected:
        print("조건에 맞는 질문이 없습니다.")
        return 2

    sys.path.insert(0, str(ROOT))
    from app.rag.tool_backend import build_search_backend  # noqa: E402  (검증 모드에서는 불러오지 않는다)

    search = build_search_backend()
    filters = {"collection": args.collection, "source_type": args.source_type}
    results = run(selected, search, k_fetch=args.k_fetch, filters=filters)

    _print_results(results)
    print("\n== 요약 ==")
    _print_summary("전체", summarize(results))
    for key, label in (("kind", "종류별"), ("category", "분야별"), ("confidence", "확신도별")):
        print(f"-- {label}")
        for name, summary in sorted(group_summary(results, key).items()):
            _print_summary(name, summary)
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n검색 오류 {len(errors)}건 (요약에서 제외됨)")

    if args.compare:
        previous = json.loads(args.compare.read_text(encoding="utf-8"))["results"]
        diff = compare(results, previous)
        print(f"\n== {args.compare.name} 대비 ==  좋아짐 {len(diff['better'])}  나빠짐 {len(diff['worse'])}")
        for label in ("better", "worse"):
            for line in diff[label]:
                print(f"  {'+' if label == 'better' else '-'} {line}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "cases_file": str(args.cases.name),
            "filters": filters,
            "k_fetch": args.k_fetch,
            "summary": summarize(results),
            "results": results,
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
