"""명령줄. 사용법은 README_적용안내.md 참고.

  python -m app.chunking.accident_ingest run      --raw <원본 폴더> --out <폴더> [--profile sif]
  python -m app.chunking.accident_ingest read     --raw <원본 폴더> --out <폴더>
  python -m app.chunking.accident_ingest map      --out <폴더> [--tiers A,B]
  python -m app.chunking.accident_ingest clean    --out <폴더>
  python -m app.chunking.accident_ingest chunk    --out <폴더> [--max-chars 1500]
  python -m app.chunking.accident_ingest fields   --out <폴더> [--no-header] [--overview-rescue]
  python -m app.chunking.accident_ingest validate --out <폴더>
  python -m app.chunking.accident_ingest show     --out <폴더> [--per-cate 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import stages
from .config import BuildSettings
from .profiles import get_profile
from .reader import ReaderError


def _add_settings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="sif")
    p.add_argument("--max-chars", type=int, default=None, help="합본이 이 글자 수를 넘을 때만 섹션으로 나눔")
    p.add_argument("--tiers", default=None, help="적재할 티어(쉼표). 기본 A,B")
    p.add_argument("--no-header", action="store_true", help="임베딩 입력에 머리말을 붙이지 않음")
    p.add_argument("--overview-rescue", action="store_true", help="전업종 accident_type이 비면 개요 키워드로 채움(오탐 주의)")


def _settings(args: argparse.Namespace) -> BuildSettings:
    kw: dict = {}
    if getattr(args, "max_chars", None) is not None:
        kw["max_chars"] = args.max_chars
    if getattr(args, "tiers", None):
        kw["include_tiers"] = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    if getattr(args, "no_header", False):
        kw["add_header"] = False
    if getattr(args, "overview_rescue", False):
        kw["overview_rescue"] = True
    return BuildSettings(**kw)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m app.chunking.accident_ingest", description="사고 데이터 원본 페이지 → 임베딩 직전 청크")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def out_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, type=Path, help="중간 파일을 둘 작업 폴더")

    p = sub.add_parser("run", help="모든 단계를 순서대로")
    p.add_argument("--raw", required=True, type=Path, help="받아 둔 API 원본 페이지(JSON) 폴더")
    out_arg(p)
    _add_settings(p)

    p = sub.add_parser("read", help="S1 원본 읽기")
    p.add_argument("--raw", required=True, type=Path)
    out_arg(p)

    for name, text in (("map", "S2 id·티어"), ("clean", "S3 정리"), ("chunk", "S4 청킹"), ("fields", "S5 부가 필드"), ("validate", "S6 검증")):
        p = sub.add_parser(name, help=text)
        out_arg(p)
        _add_settings(p)

    p = sub.add_parser("show", help="눈 확인용 표본 출력")
    out_arg(p)
    p.add_argument("--per-cate", type=int, default=10)
    return ap


def _print_report(report: dict) -> None:
    s = report["summary"]
    print(f"데이터셋: {report['dataset']}")
    print(f"레코드 {s['records']}건 {s['tiers']} → 선택 {s['selected']}건 → 청크 {s['chunks']}개")
    for c in report["checks"]:
        if c["severity"] == "info":
            mark = "INFO"
        else:
            mark = "OK  " if c["ok"] else ("FAIL" if c["severity"] == "error" else "WARN")
        print(f"  [{mark}] {c['name']}")
    print("결과:", "통과" if report["ok"] else f"실패 ({', '.join(s['errors'])})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            report = stages.run_all(args.raw, args.out, _settings(args), args.profile)
            _print_report(report)
            return 0 if report["ok"] else 1
        if args.cmd == "read":
            m = stages.stage_read(args.raw, args.out)
            print(f"읽음: 페이지 {len(m['pages'])}개, 레코드 {m['records']}건, 수집 완전성 {m['complete']}")
        elif args.cmd == "map":
            r = stages.stage_map(args.out, _settings(args), get_profile(args.profile))
            print(f"map: 티어 {r['tier_counts']} → 선택 {r['selected']}건, id 충돌 묶음 {r['id_summary']['collision_groups']}개")
        elif args.cmd == "clean":
            print(f"정리: {len(stages.stage_clean(args.out))}건")
        elif args.cmd == "chunk":
            print(f"조각: {len(stages.stage_chunk(args.out, _settings(args)))}개")
        elif args.cmd == "fields":
            records = stages.stage_fields(args.out, _settings(args), get_profile(args.profile))
            print(f"청크 레코드: {len(records)}개 → {args.out / stages.F_CHUNKS}")
        elif args.cmd == "validate":
            report = stages.stage_validate(args.out, _settings(args), get_profile(args.profile))
            _print_report(report)
            return 0 if report["ok"] else 1
        elif args.cmd == "show":
            print(stages.sample_review(args.out, args.per_cate))
            print(f"→ {args.out / stages.F_SAMPLE}")
    except (ReaderError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
