"""명령줄. 사용법은 README_적용안내.md 참고.

  python -m app.law_ingest run   <파일.hwpx> --out <폴더> [--toc 목차.txt]
  python -m app.law_ingest read  <파일.hwpx> --out <폴더>
  python -m app.law_ingest parse    --out <폴더>
  python -m app.law_ingest clean    --out <폴더>
  python -m app.law_ingest chunk    --out <폴더> [--max-chars 1000]
  python -m app.law_ingest fields   --out <폴더> [--no-header] [--tags] [--year-basis effective]
  python -m app.law_ingest validate --out <폴더> [--toc 목차.txt]
  python -m app.law_ingest diff     --out <새 폴더> --prev <이전 폴더>      (개정 때: 올릴 것·지울 것 계산)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stages
from .config import BuildSettings
from .parse import ParseError
from .readers import ReaderError


def _add_settings(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-chars", type=int, default=None, help="조를 나누는 글자 수 상한")
    p.add_argument("--no-header", action="store_true", help="임베딩 입력에 계층 머리말을 붙이지 않음")
    p.add_argument("--tags", action="store_true", help="계층 경로로 hazard/work_type 태그를 붙임(초안)")
    p.add_argument("--year-basis", choices=["promulgation", "effective"], default=None)
    p.add_argument("--law-key", default=None)
    p.add_argument("--file-path", default=None, help="kb_sources.file_path에 쓸 값")
    p.add_argument("--url", default=None)


def _add_prev(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prev", type=Path, default=None, help="이전 판 작업 폴더(있으면 끝에 갱신 계획을 만든다)")
    p.add_argument("--allow-older", action="store_true", help="이전보다 시행일이 앞선 파일도 허용")
    p.add_argument("--allow-mass-remove", action="store_true", help="지울 청크가 이전의 10%를 넘어도 허용")


def _levels(args: argparse.Namespace) -> tuple[str, ...] | None:
    raw = getattr(args, "heading_levels", None)
    return tuple(x.strip() for x in raw.split(",") if x.strip()) if raw else None


def _settings(args: argparse.Namespace) -> BuildSettings:
    kw: dict = {}
    if getattr(args, "max_chars", None) is not None:
        kw["max_chars"] = args.max_chars
    if getattr(args, "no_header", False):
        kw["add_header"] = False
    if getattr(args, "tags", False):
        kw["tags_enabled"] = True
    if getattr(args, "year_basis", None):
        kw["year_basis"] = args.year_basis
    for name in ("law_key", "file_path", "url"):
        if getattr(args, name, None):
            kw[name] = getattr(args, name)
    return BuildSettings(**kw)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m app.law_ingest", description="법령 파일 → 임베딩 직전 청크")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def out_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", required=True, type=Path, help="중간 파일을 둘 작업 폴더")

    p = sub.add_parser("run", help="모든 단계를 순서대로")
    p.add_argument("input", type=Path)
    out_arg(p)
    p.add_argument("--toc", type=Path, default=None, help="목차 txt(조 목록 대조용, 선택)")
    p.add_argument("--expect-articles", type=int, default=None)
    p.add_argument("--heading-levels", default=None, help="제목 단계(위→아래, 쉼표). 기본 편,장,절,관,속")
    _add_prev(p)
    _add_settings(p)

    p = sub.add_parser("read", help="S0·S1 원본 읽기")
    p.add_argument("input", type=Path)
    out_arg(p)

    p = sub.add_parser("parse", help="S2 구조 파싱")
    out_arg(p)
    p.add_argument("--heading-levels", default=None, help="제목 단계(위→아래, 쉼표). 기본 편,장,절,관,속")

    p = sub.add_parser("clean", help="S3 정리")
    out_arg(p)

    p = sub.add_parser("chunk", help="S4 청킹")
    out_arg(p)
    _add_settings(p)

    p = sub.add_parser("fields", help="S5 부가 필드")
    out_arg(p)
    _add_settings(p)

    p = sub.add_parser("validate", help="S6 검증")
    out_arg(p)
    p.add_argument("--toc", type=Path, default=None)
    p.add_argument("--expect-articles", type=int, default=None)

    p = sub.add_parser("diff", help="S7 이전 판과 비교(갱신 계획)")
    out_arg(p)
    p.add_argument("--prev", required=True, type=Path, help="이전 판 작업 폴더")
    p.add_argument("--allow-older", action="store_true")
    p.add_argument("--allow-mass-remove", action="store_true")
    return ap


def _print_report(report: dict) -> None:
    s = report["summary"]
    print(f"법령: {report['law']}")
    print(f"조 {s['articles']}개 {s['by_status']} / 청크 {s['chunks']}개 / 나뉜 조 {s['split_articles']}개")
    for c in report["checks"]:
        mark = "OK  " if c["ok"] else ("FAIL" if c["severity"] == "error" else "WARN")
        print(f"  [{mark}] {c['name']}")
    print("결과:", "통과" if report["ok"] else f"실패 ({', '.join(s['errors'])})")
    if "update_plan" in report:
        _print_plan(report["update_plan"])


def _print_plan(s: dict) -> None:
    print(f"갱신 계획: 추가 {s['add']} / 변경 {s['update']} / 삭제 {s['remove']} / 그대로 {s['unchanged']}  (이전 {s['prev_chunks']} → 이번 {s['new_chunks']})")
    if s["remove_reasons"]:
        print("  삭제 사유:", s["remove_reasons"])
    for note in s["needs_review"]:
        print("  확인 필요:", note)
    if s["same_source_file"]:
        print("  참고: 원본 파일이 이전과 같습니다.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            report = stages.run_all(
                args.input,
                args.out,
                _settings(args),
                toc=args.toc,
                expect_articles=args.expect_articles,
                levels=_levels(args),
                prev=args.prev,
                allow_older=args.allow_older,
                allow_mass_remove=args.allow_mass_remove,
            )
            _print_report(report)
            return 0 if report["ok"] else 1
        if args.cmd == "read":
            doc = stages.stage_read(args.input, args.out)
            print(f"읽음: 문단 {len(doc.paragraphs)}개, {doc.meta.law_name}, 시행 {doc.meta.effective_date}, {doc.meta.promulgation_no}")
        elif args.cmd == "parse":
            law = stages.stage_parse(args.out, _levels(args))
            print(f"파싱: 제목 {len(law.headings)}개, 조 {len(law.articles)}개, 부칙 {len(law.addenda)}줄, 경고 {len(law.warnings)}건")
        elif args.cmd == "clean":
            law = stages.stage_clean(args.out)
            by: dict[str, int] = {}
            for a in law.articles:
                by[a.status] = by.get(a.status, 0) + 1
            print(f"정리: {by}")
        elif args.cmd == "chunk":
            pieces = stages.stage_chunk(args.out, _settings(args))
            print(f"조각: {len(pieces)}개")
        elif args.cmd == "fields":
            records = stages.stage_fields(args.out, _settings(args))
            print(f"청크 레코드: {len(records)}개 → {args.out / stages.F_CHUNKS}")
        elif args.cmd == "validate":
            report = stages.stage_validate(args.out, toc=args.toc, expect_articles=args.expect_articles)
            _print_report(report)
            return 0 if report["ok"] else 1
        elif args.cmd == "diff":
            plan = stages.stage_diff(args.out, args.prev, allow_older=args.allow_older, allow_mass_remove=args.allow_mass_remove)
            _print_plan(plan["summary"])
            print(f"→ {args.out / stages.F_CHANGES}, {args.out / stages.F_UPSERT}")
    except (ReaderError, ParseError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
