"""RAG 검색 도구를 서버 없이 시험한다: 어댑터 → 규격 검사 → 결과 출력.

    python scripts/try_rag_tools.py                                  # 내장 시험용 backend(실제 RAG 아님)로 배선만 확인
    python scripts/try_rag_tools.py --backend app.rag.tool_backend:build_search_backend
    python scripts/try_rag_tools.py --backend ... --k 3 --min-score 0.3 "배관 용접 화재" "탱크 내부 질식"
    (환경변수 RAG_BACKEND가 있으면 --backend 생략 가능)

프로젝트 루트에서 실행한다. 도구마다 소요 시간, 규격 위반, 상위 결과를 출력한다. 위반이 있으면 종료 코드 1.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.exceptions import ToolError  # noqa: E402
from app.tools.adapters.rag_tools import register_rag_tools  # noqa: E402
from app.tools.contract_check import check_search_output  # noqa: E402
from app.tools.rag_loader import load_search_backend  # noqa: E402

DEFAULT_QUERIES = ["배관 용접 중 불티로 인한 화재", "탱크 내부 작업 중 질식", "지붕 위 작업 중 추락"]
_NS = uuid.UUID("00000000-0000-0000-0000-00000000d3a0")
_DEMO_DOCS = [  # 시험용 가짜 자료. 실제 RAG 결과가 아니다
    ("demo-1", "[demo] 사고사례", "배관 용접 불티가 가연물에 옮겨붙어 화재"),
    ("demo-2", "[demo] 사고사례", "탱크 내부 산소 결핍 질식"),
    ("demo-3", "[demo] 지침", "고소작업 안전대 착용 추락 방지"),
]


def demo_backend(query, *, k, filters):
    words = set(query.split())
    hits = [
        {"id": str(uuid.uuid5(_NS, d)), "doc_id": d, "source": s, "text": t, "score": len(words & set(t.split())) / max(len(words), 1)}
        for d, s, t in _DEMO_DOCS
    ]
    return sorted(hits, key=lambda h: -h["score"])[:k]


class Collector:
    """ToolRegistry 대신 등록된 함수를 모은다 (실제 레지스트리 없이도 시험하려고)."""

    def __init__(self):
        self.tools = {}

    def register(self, name, function, **_kwargs):
        self.tools[name] = function


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="*", help="검색 문장 (없으면 내장 3개)")
    parser.add_argument("--backend", default=os.environ.get("RAG_BACKEND"), help="모듈경로:함수이름 (생략하면 내장 시험용)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    loaded = load_search_backend(args.backend)
    if loaded is None:
        print("※ RAG_BACKEND가 없어 내장 시험용 backend를 씁니다 (배선 확인용, 실제 RAG 아님)\n")
        search, errors = demo_backend, ()
    else:
        search, errors = loaded.search, loaded.errors
        print(f"backend: {args.backend}\n")

    collector = Collector()
    register_rag_tools(collector, search, backend_errors=errors)
    failed = False
    for query in args.queries or DEFAULT_QUERIES:
        for name, function in collector.tools.items():
            arguments = {"query": query, "k": args.k}
            if args.min_score is not None:
                arguments["min_score"] = args.min_score
            started = time.perf_counter()
            try:
                output = function(arguments)
            except ToolError as exc:
                print(f"[{name}] {query!r} → ToolError {exc.code}: {exc.message}")
                failed = True
                continue
            ms = int((time.perf_counter() - started) * 1000)
            problems = check_search_output(output, k=args.k, min_score=args.min_score)
            failed |= bool(problems)
            print(f"[{name}] {query!r} → {len(output['hits'])}건, {ms}ms, 규격 {'위반 ' + str(problems) if problems else '적합'}")
            for hit in output["hits"][:3]:
                print(f"    {hit['score']:.3f}  {hit['source']}  {hit['text'][:60]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
