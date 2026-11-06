"""단계별 실행과 중간 파일.

작업 폴더(--out) 안의 파일:
  manifest.json       S0  원본 정보(sha256, 시행일, 공포번호 …)
  raw/<원본>          S0  원본 사본
  01_source.json      S1  읽기 결과(메타 + 문단)
  02_parsed.json      S2  구조 파싱 결과
  03_cleaned.json     S3  정리 결과(+ 글자 보존 장부)
  04_pieces.json      S4  조각
  05_chunks.jsonl     S5  임베딩 직전 청크(한 줄에 청크 하나)
  06_validation.json  S6  검증 보고서
  07_changes.json     S7  이전 판과의 비교(add / update / remove). --prev를 줄 때만 만든다
  07_upsert.jsonl     S7  새로 올리거나 덮어쓸 청크(add + update)

각 단계는 앞 단계의 파일만 읽는다. 어느 단계든 파일을 열어 눈으로 확인하고 그 단계부터 다시 돌릴 수 있다.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk import make_pieces
from .clean import clean_law
from .config import BuildSettings
from .fields import build_records
from .models import SCHEMA_VERSION, ParsedLaw, Piece, SourceDoc
from .parse import parse_source
from .readers import read_source
from .update import key_of, plan_update
from .validate import validate_all

F_MANIFEST = "manifest.json"
F_SOURCE = "01_source.json"
F_PARSED = "02_parsed.json"
F_CLEANED = "03_cleaned.json"
F_PIECES = "04_pieces.json"
F_CHUNKS = "05_chunks.jsonl"
F_VALIDATION = "06_validation.json"
F_CHANGES = "07_changes.json"
F_UPSERT = "07_upsert.jsonl"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"앞 단계 결과 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_read(input_path: str | Path, out: Path) -> SourceDoc:
    input_path = Path(input_path)
    doc = read_source(input_path)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, raw_dir / input_path.name)
    _write_json(
        out / F_MANIFEST,
        {
            "schema_version": SCHEMA_VERSION,
            "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_copy": f"raw/{input_path.name}",
            **doc.meta.__dict__,
            "paragraph_count": len(doc.paragraphs),
        },
    )
    _write_json(out / F_SOURCE, doc.to_dict())
    return doc


def load_source(out: Path) -> SourceDoc:
    return SourceDoc.from_dict(_read_json(out / F_SOURCE))


def stage_parse(out: Path, levels: tuple[str, ...] | None = None) -> ParsedLaw:
    law = parse_source(load_source(out), levels)
    _write_json(out / F_PARSED, law.to_dict())
    return law


def load_parsed(out: Path) -> ParsedLaw:
    return ParsedLaw.from_dict(_read_json(out / F_PARSED))


def stage_clean(out: Path) -> ParsedLaw:
    law = clean_law(load_parsed(out))
    _write_json(out / F_CLEANED, law.to_dict())
    return law


def load_cleaned(out: Path) -> ParsedLaw:
    return ParsedLaw.from_dict(_read_json(out / F_CLEANED))


def stage_chunk(out: Path, settings: BuildSettings) -> list[Piece]:
    pieces = make_pieces(load_cleaned(out), settings.max_chars)
    _write_json(out / F_PIECES, {"max_chars": settings.max_chars, "pieces": [p.__dict__ for p in pieces]})
    return pieces


def load_pieces(out: Path) -> list[Piece]:
    return [Piece.from_dict(d) for d in _read_json(out / F_PIECES)["pieces"]]


def stage_fields(out: Path, settings: BuildSettings) -> list[dict[str, Any]]:
    records = build_records(load_cleaned(out), load_pieces(out), settings)
    path = out / F_CHUNKS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def load_chunks(out: Path) -> list[dict[str, Any]]:
    path = out / F_CHUNKS
    if not path.exists():
        raise FileNotFoundError(f"앞 단계 결과 파일이 없습니다: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_validate(out: Path, toc: str | Path | None = None, expect_articles: int | None = None) -> dict[str, Any]:
    toc_lines = None
    if toc is not None:
        toc_lines = Path(toc).read_text(encoding="utf-8").splitlines()
    report = validate_all(
        load_source(out),
        load_parsed(out),
        load_cleaned(out),
        load_chunks(out),
        toc_lines=toc_lines,
        expect_articles=expect_articles,
    )
    _write_json(out / F_VALIDATION, report)
    return report


def stage_diff(out: Path, prev: Path, *, allow_older: bool = False, allow_mass_remove: bool = False) -> dict[str, Any]:
    """이전 작업 폴더(prev)와 이번 청크를 비교한다. 이번 검증이 통과하지 않았으면 거부한다."""
    report = _read_json(out / F_VALIDATION)
    if not report.get("ok"):
        raise ValueError("이번 검증이 통과하지 않아 비교하지 않습니다. 먼저 validate를 통과시키세요.")
    prev_chunks = load_chunks(prev)
    prev_report = prev / F_VALIDATION
    if prev_report.exists() and not json.loads(prev_report.read_text(encoding="utf-8")).get("ok"):
        raise ValueError("이전 작업 폴더의 검증이 통과 상태가 아닙니다. 이전 판을 믿을 수 없어 비교하지 않습니다.")
    new_chunks = load_chunks(out)
    statuses = {a.key: a.status for a in load_cleaned(out).articles}
    plan = plan_update(prev_chunks, new_chunks, statuses, allow_older=allow_older, allow_mass_remove=allow_mass_remove)
    _write_json(out / F_CHANGES, plan)
    wanted = {key_of_entry(e) for e in (*plan["add"], *plan["update"])}
    with (out / F_UPSERT).open("w", encoding="utf-8", newline="\n") as f:
        for r in new_chunks:
            if key_of(r) in wanted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return plan


def key_of_entry(entry: dict[str, Any]) -> tuple[str, str, int]:
    nk = entry["natural_key"]
    return nk["law_key"], nk["article_key"], int(nk["part"])


def run_all(
    input_path: str | Path,
    out: Path,
    settings: BuildSettings,
    toc: str | Path | None = None,
    expect_articles: int | None = None,
    levels: tuple[str, ...] | None = None,
    prev: Path | None = None,
    allow_older: bool = False,
    allow_mass_remove: bool = False,
) -> dict[str, Any]:
    stage_read(input_path, out)
    stage_parse(out, levels)
    stage_clean(out)
    stage_chunk(out, settings)
    stage_fields(out, settings)
    report = stage_validate(out, toc=toc, expect_articles=expect_articles)
    if prev is not None and report["ok"]:
        report["update_plan"] = stage_diff(out, prev, allow_older=allow_older, allow_mass_remove=allow_mass_remove)["summary"]
    return report
