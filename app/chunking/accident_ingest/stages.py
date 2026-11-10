"""단계별 실행과 중간 파일.

작업 폴더(--out) 안의 파일:
  raw/<페이지>.json      S0  받아 둔 원본 페이지 사본(읽을 때 복사)
  manifest.json          S1  페이지별 sha256·건수, 업종별 완전성(complete)
  01_records.json        S1  읽기 결과(레코드, 값은 원본 그대로 + 위치)
  02_docs.json           S2  external_id·티어·선택 여부를 붙인 전체 레코드(보류한 티어도 남는다)
  03_cleaned.json        S3  선택된 사례의 정리 결과(+ 글자 보존 장부)
  04_pieces.json         S4  조각
  05_chunks.jsonl        S5  임베딩 직전 청크(한 줄에 청크 하나)
  05_tag_report.json     S5  업종별 태그 채움 비율·값 분포
  06_validation.json     S6  검증 보고서
  07_sample_review.md    눈 확인용 표본(show)

각 단계는 앞 단계의 파일만 읽는다. 어느 단계든 파일을 열어 눈으로 확인하고 그 단계부터 다시 돌릴 수 있다.
S7(이전 판과 비교)과 S8(적재)은 이번 범위가 아니다.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from . import ids
from .chunk import make_pieces
from .clean import clean_doc
from .config import BuildSettings
from .fields import build_records, tag_report
from .models import SCHEMA_VERSION, sha256_json
from .profiles import get_profile
from .reader import read_pages
from .validate import validate_all

F_MANIFEST = "manifest.json"
F_RECORDS = "01_records.json"
F_DOCS = "02_docs.json"
F_CLEANED = "03_cleaned.json"
F_PIECES = "04_pieces.json"
F_CHUNKS = "05_chunks.jsonl"
F_TAG_REPORT = "05_tag_report.json"
F_VALIDATION = "06_validation.json"
F_SAMPLE = "07_sample_review.md"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"앞 단계 결과 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_read(raw_dir: str | Path, out: Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    records, manifest = read_pages(raw_dir)
    dest = out / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    for p in manifest["pages"]:
        shutil.copyfile(raw_dir / p["file"], dest / p["file"])
    manifest = {"schema_version": SCHEMA_VERSION, "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **manifest}
    _write_json(out / F_MANIFEST, manifest)
    _write_json(out / F_RECORDS, {"records": records})
    return manifest


def stage_map(out: Path, settings: BuildSettings, profile: ModuleType) -> dict[str, Any]:
    records = _read_json(out / F_RECORDS)["records"]
    docs = []
    for r in records:
        d = dict(r)
        d["content_sha256"] = sha256_json({k: v for k, v in r.items() if not k.startswith("_")})
        d["tier"] = profile.tier_of(d)
        docs.append(d)
    id_summary = ids.assign_ids(docs, profile.ID_PREFIX)
    for d in docs:
        d["selected"] = d["tier"] in settings.include_tiers
    tiers: dict[str, int] = {}
    for d in docs:
        tiers[d["tier"]] = tiers.get(d["tier"], 0) + 1
    result = {
        "profile": profile.NAME,
        "include_tiers": list(settings.include_tiers),
        "tier_counts": dict(sorted(tiers.items())),
        "selected": sum(1 for d in docs if d["selected"]),
        "id_summary": id_summary,
        "docs": docs,
    }
    _write_json(out / F_DOCS, result)
    return result


def load_docs(out: Path) -> dict[str, Any]:
    return _read_json(out / F_DOCS)


def stage_clean(out: Path) -> list[dict[str, Any]]:
    cleaned = [clean_doc(d) for d in load_docs(out)["docs"] if d["selected"]]
    _write_json(out / F_CLEANED, {"docs": cleaned})
    return cleaned


def load_cleaned(out: Path) -> list[dict[str, Any]]:
    return _read_json(out / F_CLEANED)["docs"]


def stage_chunk(out: Path, settings: BuildSettings) -> list[dict[str, Any]]:
    cleaned = load_cleaned(out)
    pieces: list[dict[str, Any]] = []
    no_narrative: list[str] = []
    for d in cleaned:
        ps = make_pieces(d, settings)
        if not ps:
            no_narrative.append(d["external_id"])
        pieces.extend(ps)
    _write_json(out / F_PIECES, {"max_chars": settings.max_chars, "no_narrative": no_narrative, "pieces": pieces})
    return pieces


def load_pieces(out: Path) -> list[dict[str, Any]]:
    return _read_json(out / F_PIECES)["pieces"]


def stage_fields(out: Path, settings: BuildSettings, profile: ModuleType) -> list[dict[str, Any]]:
    cleaned = load_cleaned(out)
    records = build_records(cleaned, load_pieces(out), settings, profile)
    path = out / F_CHUNKS
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_json(out / F_TAG_REPORT, tag_report(records, cleaned))
    return records


def load_chunks(out: Path) -> list[dict[str, Any]]:
    path = out / F_CHUNKS
    if not path.exists():
        raise FileNotFoundError(f"앞 단계 결과 파일이 없습니다: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_validate(out: Path, settings: BuildSettings, profile: ModuleType) -> dict[str, Any]:
    report = validate_all(
        _read_json(out / F_MANIFEST),
        load_docs(out),
        load_cleaned(out),
        load_pieces(out),
        load_chunks(out),
        settings,
        profile,
    )
    _write_json(out / F_VALIDATION, report)
    return report


def run_all(raw_dir: str | Path, out: Path, settings: BuildSettings, profile_name: str = "sif") -> dict[str, Any]:
    profile = get_profile(profile_name)
    stage_read(raw_dir, out)
    stage_map(out, settings, profile)
    stage_clean(out)
    stage_chunk(out, settings)
    stage_fields(out, settings, profile)
    return stage_validate(out, settings, profile)


def sample_review(out: Path, per_cate: int = 10) -> str:
    """업종마다 고르게 떨어진 청크 per_cate개를 눈으로 읽기 좋게 만든다."""
    chunks = load_chunks(out)
    lines: list[str] = ["# 청크 눈 확인 표본", ""]
    for cate in ("건설업", "전업종"):
        group = [c for c in chunks if c["cate"] == cate]
        if not group:
            continue
        k = min(per_cate, len(group))
        picks = [group[(i * len(group)) // k] for i in range(k)]
        lines += [f"## {cate} (전체 {len(group)}청크 중 {k}개)", ""]
        for c in picks:
            lines += [
                f"### {c['title']}",
                f"- id `{c['external_id']}`  티어 {c['tier']}  chunk_index {c['chunk_index']}  flagged {c['flagged']}",
                f"- 머리말: {c['header']}",
                f"- locator: {c['locator']}",
                f"- 태그: severity={c['severity']}  accident_type={c['accident_type']}  hazard={c['hazard']}  work_type={c['work_type']}  year={c['year']}",
                f"- 원본 분류: situation={c['situation']}  disasterType={c['disaster_type']}",
                "",
                "```",
                c["text"],
                "```",
                "",
            ]
    text = "\n".join(lines)
    (out / F_SAMPLE).write_text(text, encoding="utf-8", newline="\n")
    return text
