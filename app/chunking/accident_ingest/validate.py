"""S6 검증: 청크 파일을 확정해도 되는지 확인한다. 하나라도 error면 확정하지 않는다.

수집
  fetch_complete         업종별 페이지가 1..N 연속이고 건수 합 = totalCount
  result_code_ok         읽은 모든 페이지의 resultCode == "00"
  cate_allowed           레코드의 업종 ∈ 허용 목록
식별·범위
  record_ids             전체 레코드에 external_id, 유일, 1~100자, ':'·공백 없음
  id_stability           정규화·해시 golden (적재 후 바뀌면 안 되는 값)
  id_collision           개요 충돌 쌍(전체 기준). 적재 대상 안에 있으면 경고
  tier_counts            A+B+C+X = 전체, 선택 수 = 포함 티어 수
글자 보존
  field_conservation     서술 필드 raw = kept + markup + entities + control, kept = 정리 결과
  kept_matches_chunks    정리된 서술 글자 수 = 청크로 나간 글자 수(되풀이한 줄 제외)
  split_reconstruction   조각을 이으면(되풀이한 줄 제외) 합본 원문과 같음
  chunk_coverage         서술이 있는 사례는 청크가 있음
청크
  chunk_fields, natural_keys_unique, no_noise, embed_text_length, chunk_oversize, request_limits,
  payload_types, field_len_at_max, profile_worst_case, chunks_per_source, unknown_fields, empty_sources
태그
  tag_codes_valid, cas_checksum, year_range, tag_coverage(info), requirements_coverage(info)
기타
  pii_scan, flagged_count(info), golden
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from types import ModuleType
from typing import Any

from . import config, golden, ids
from .chunk import case_text, make_pieces, sections_of
from .clean import NARRATIVE_FIELDS, clean_text, is_empty
from .config import BuildSettings
from .models import nonspace

_PII_RES = {
    "주민등록번호": re.compile(r"\b\d{6}-[1-4]\d{6}\b"),
    "전화번호": re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b"),
    "이메일": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
}
_CAS_RE = re.compile(r"^[0-9]{2,7}-[0-9]{2}-[0-9]$")


def _check(name: str, ok: bool, detail: Any = None, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "detail": detail}


def _cas_ok(value: str) -> bool:
    if not _CAS_RE.match(value):
        return False
    digits = value.replace("-", "")
    body, check = digits[:-1], int(digits[-1])
    total = sum(int(d) * i for i, d in enumerate(reversed(body), start=1))
    return total % 10 == check


def _squash(text: str) -> str:
    return "".join(text.split())


def _worst_case_pieces(settings: BuildSettings) -> int:
    """가이드 최대 필드 크기(2,000/1,000/4,000자)의 줄 단위 본문으로 만든 조각 수."""
    def lines(n: int) -> str:
        return "\n".join("▶ " + "가" * 38 for _ in range(max(1, n // 40)))

    clean = {field: lines(config.FIELD_GUIDE_MAX[field]) for _, field, _ in config.SECTIONS}
    return len(make_pieces({"external_id": "WORST", "clean": clean}, settings))


def validate_all(
    manifest: dict[str, Any],
    docs_file: dict[str, Any],
    cleaned: list[dict[str, Any]],
    pieces: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    settings: BuildSettings,
    profile: ModuleType,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    docs = docs_file["docs"]
    year_max = settings.year_max or date.today().year

    # --- 수집 -------------------------------------------------------------------------
    checks.append(_check("fetch_complete", manifest.get("complete") is True, manifest.get("groups")))
    bad_pages = [p["file"] for p in manifest.get("pages", []) if p.get("result_code") != "00"]
    checks.append(_check("result_code_ok", not bad_pages, {"pages": len(manifest.get("pages", [])), "bad": bad_pages}))
    bad_cate = sorted({d["cateSeNm"] for d in docs if d["cateSeNm"] not in profile.CATE_ALLOWED})
    checks.append(_check("cate_allowed", not bad_cate, {"unexpected": bad_cate}))

    # --- 식별·범위 --------------------------------------------------------------------
    all_ids = [d["external_id"] for d in docs]
    cnt = Counter(all_ids)
    dup_ids = sorted(k for k, v in cnt.items() if v > 1)
    bad_ids = [i for i in all_ids if not (1 <= len(i) <= config.MAX_EXTERNAL_ID_CHARS) or ":" in i or any(c.isspace() for c in i)]
    checks.append(_check("record_ids", not dup_ids and not bad_ids, {"records": len(all_ids), "duplicates": dup_ids[:10], "invalid": bad_ids[:10]}))

    bad_gold = [(src, want, ids.base_id(src, profile.ID_PREFIX)) for src, want in golden.ID_GOLDEN if ids.base_id(src, profile.ID_PREFIX) != want]
    checks.append(_check("id_stability", not bad_gold, {"cases": len(golden.ID_GOLDEN), "mismatch": bad_gold}))

    collisions = docs_file.get("id_summary", {}).get("collisions", [])
    selected_ids = {d["external_id"] for d in docs if d["selected"]}
    in_scope = [c for c in collisions if any(i in selected_ids for i in c["external_ids"])]
    checks.append(
        _check(
            "id_collision",
            not in_scope,
            {"groups_total": len(collisions), "groups_in_scope": len(in_scope), "in_scope": in_scope[:5]},
            severity="warn",
        )
    )

    tier_counts = Counter(d["tier"] for d in docs)
    want_selected = sum(tier_counts[t] for t in settings.include_tiers)
    got_selected = sum(1 for d in docs if d["selected"])
    checks.append(
        _check(
            "tier_counts",
            sum(tier_counts.values()) == len(docs) and got_selected == want_selected,
            {"tiers": dict(sorted(tier_counts.items())), "include": list(settings.include_tiers), "selected": got_selected},
        )
    )

    # --- 글자 보존 --------------------------------------------------------------------
    cons_bad: list[str] = []
    for d in cleaned:
        for f in NARRATIVE_FIELDS:
            led = d["ledger"][f]
            text, again = clean_text(d["raw"][f])
            expected_clean = "" if is_empty(text) else text
            ok = (
                led["raw"] == led["kept"] + led["markup"] + led["entities"] + led["control"]
                and led["raw"] == nonspace(d["raw"][f])
                and again == led
                and led["kept"] == nonspace(text)
                and d["clean"][f] == expected_clean
            )
            if not ok:
                cons_bad.append(f"{d['external_id']}:{f}")
    checks.append(_check("field_conservation", not cons_bad, {"docs": len(cleaned), "bad": cons_bad[:10]}))

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for p in pieces:
        by_doc.setdefault(p["external_id"], []).append(p)

    kept_bad: list[str] = []
    split_bad: list[str] = []
    no_chunk: list[str] = []
    for d in cleaned:
        secs = sections_of(d["clean"])
        ps = by_doc.get(d["external_id"], [])
        if not secs:
            continue
        if not ps:
            no_chunk.append(d["external_id"])
            continue
        expected = _squash(case_text(secs))
        got = _squash("\n".join("\n".join(p["lines"][p["repeated_lead_lines"]:]) for p in ps))
        if len(got) != len(expected):
            kept_bad.append(d["external_id"])
        if got != expected:
            split_bad.append(d["external_id"])
    checks.append(_check("kept_matches_chunks", not kept_bad, {"bad": kept_bad[:10]}))
    checks.append(_check("split_reconstruction", not split_bad, {"split_docs": sum(1 for v in by_doc.values() if any(p["section"] != "case" for p in v)), "bad": split_bad[:10]}))
    checks.append(_check("chunk_coverage", not no_chunk, {"without_chunks": no_chunk[:10]}))

    empty_docs = [d["external_id"] for d in cleaned if not sections_of(d["clean"])]
    checks.append(_check("empty_sources", not empty_docs, {"count": len(empty_docs), "ids": empty_docs[:10]}, severity="warn"))

    # --- 청크 -------------------------------------------------------------------------
    need = ("text", "embed_text", "title", "locator", "external_id", "source_type", "collection")
    bad_fields = [r["natural_key"] for r in chunks if any(not r.get(k) for k in need)]
    checks.append(_check("chunk_fields", not bad_fields, {"chunks": len(chunks), "bad": bad_fields[:10]}))

    nk = Counter(tuple(r["natural_key"].values()) for r in chunks)
    ci = Counter((r["external_id"], r["chunk_index"]) for r in chunks)
    checks.append(
        _check("natural_keys_unique", all(v == 1 for v in nk.values()) and all(v == 1 for v in ci.values()), {"dup_natural": [k for k, v in nk.items() if v > 1][:5], "dup_chunk_index": [k for k, v in ci.items() if v > 1][:5]})
    )

    noisy = [r["natural_key"] for r in chunks if any(n in r["text"] for n in config.NOISE_STRINGS)]
    checks.append(_check("no_noise", not noisy, {"bad": noisy[:10]}))

    long_embed = [r["external_id"] for r in chunks if len(r["embed_text"]) > config.MAX_EMBED_CHARS]
    checks.append(_check("embed_text_length", not long_embed, {"limit": config.MAX_EMBED_CHARS, "bad": long_embed[:10]}, severity="warn"))
    over = [r["external_id"] for r in chunks if r["oversize"]]
    checks.append(_check("chunk_oversize", not over, {"count": len(over), "ids": over[:10]}, severity="warn"))

    lim: list[str] = []
    for r in chunks:
        if (
            len(r["title"]) > config.MAX_TITLE_CHARS
            or len(r["locator"]) > config.MAX_LOCATOR_CHARS
            or len(r["external_id"]) > config.MAX_EXTERNAL_ID_CHARS
            or len(r["license"]) > 50
            or len(r["publisher"]) > 100
            or len(r["url"]) > 500
            or len(r["text"]) > config.MAX_TEXT_CHARS
            or len(r["embed_text"]) > config.MAX_TEXT_CHARS
            or not re.fullmatch(r"[0-9a-f]{64}", r["content_sha256"])
        ):
            lim.append(r["external_id"])
    checks.append(_check("request_limits", not lim, {"bad": lim[:10]}))

    pay_bad: list[str] = []
    for r in chunks:
        y, s = r["year"], r["severity"]
        ok = (y is None or (isinstance(y, int) and not isinstance(y, bool))) and (s is None or isinstance(s, str))
        ok = ok and all(isinstance(r[k], list) and all(isinstance(x, str) for x in r[k]) for k in ("accident_type", "hazard", "work_type", "cas", "ghs"))
        ok = ok and isinstance(r["flagged"], bool) and r["site_id"] == config.SITE_ID
        if not ok:
            pay_bad.append(r["external_id"])
    checks.append(_check("payload_types", not pay_bad, {"bad": pay_bad[:10]}))

    at_max = Counter()
    for d in cleaned:
        for f, limit in config.FIELD_GUIDE_MAX.items():
            if len(d["raw"][f] or "") >= limit:
                at_max[f] += 1
    checks.append(_check("field_len_at_max", not at_max, dict(at_max), severity="warn"))

    worst = _worst_case_pieces(settings)
    checks.append(_check("profile_worst_case", worst <= config.MAX_CHUNKS_PER_SOURCE, {"worst_case_pieces": worst, "limit": config.MAX_CHUNKS_PER_SOURCE}))

    per_source = Counter(r["external_id"] for r in chunks)
    dist = Counter(per_source.values())
    over_limit = [k for k, v in per_source.items() if v > config.MAX_CHUNKS_PER_SOURCE]
    checks.append(_check("chunks_per_source", not over_limit, {"distribution": dict(sorted(dist.items())), "over_limit": over_limit[:10]}))

    unknown = sorted({k for d in cleaned for k in d["unknown_fields"]})
    checks.append(_check("unknown_fields", not unknown, {"fields": unknown}, severity="warn"))

    # --- 태그 -------------------------------------------------------------------------
    codes = profile.TAG_CODES
    tag_bad: list[dict[str, Any]] = []
    for r in chunks:
        for axis, key in (("ACCIDENT_TYPE", "accident_type"), ("HAZARD", "hazard"), ("WORK_TYPE", "work_type")):
            for v in r[key]:
                if v not in codes[axis]:
                    tag_bad.append({"id": r["external_id"], "axis": axis, "value": v})
        if r["severity"] is not None and r["severity"] not in codes["SEVERITY"]:
            tag_bad.append({"id": r["external_id"], "axis": "SEVERITY", "value": r["severity"]})
    checks.append(_check("tag_codes_valid", not tag_bad, {"bad": tag_bad[:10]}))

    cas_bad = [v for r in chunks for v in r["cas"] if not _cas_ok(v)]
    checks.append(_check("cas_checksum", not cas_bad, {"bad": cas_bad[:10]}))

    years = [r["year"] for r in chunks if r["year"] is not None]
    bad_years = [y for y in years if not (config.YEAR_MIN <= y <= year_max)]
    checks.append(_check("year_range", not bad_years, {"no_year": len(chunks) - len(years), "bad": bad_years[:10]}))

    cov: dict[str, dict[str, Any]] = {}
    for r in chunks:
        c = cov.setdefault(r["cate"], {"chunks": 0})
        c["chunks"] += 1
        for axis in ("severity", "accident_type", "hazard", "work_type", "year"):
            v = r[axis]
            if (bool(v) if isinstance(v, list) else v is not None):
                c[axis] = c.get(axis, 0) + 1
    checks.append(_check("tag_coverage", True, cov, severity="info"))

    req: dict[str, dict[str, int]] = {}
    for r in chunks:
        g = req.setdefault(r["cate"], Counter())
        for code in r["work_type"]:
            g[f"work_type={code}"] += 1
        for code in r["accident_type"]:
            g[f"accident_type={code}"] += 1
    checks.append(_check("requirements_coverage", True, {k: dict(sorted(v.items())) for k, v in req.items()}, severity="info"))

    # --- 기타 -------------------------------------------------------------------------
    pii: list[dict[str, str]] = []
    for r in chunks:
        for name, rx in _PII_RES.items():
            if rx.search(r["text"]):
                pii.append({"id": r["external_id"], "kind": name})
    checks.append(_check("pii_scan", not pii, {"hits": pii[:10]}))

    flagged = sorted({r["external_id"] for r in chunks if r["flagged"]})
    checks.append(_check("flagged_count", True, {"sources": len(flagged), "ids": flagged[:10]}, severity="info"))

    gold_bad: list[dict[str, Any]] = []
    gold_found = 0
    for g in golden.CASE_GOLDEN:
        hit = [r for r in chunks if r["text"].find(g["overview_prefix"]) >= 0]
        if not hit:
            continue
        gold_found += 1
        r = hit[0]
        for k, v in g["expect"].items():
            if r[k] != v:
                gold_bad.append({"overview_prefix": g["overview_prefix"], "field": k, "want": v, "got": r[k]})
    checks.append(_check("golden", not gold_bad, {"defined": len(golden.CASE_GOLDEN), "found": gold_found, "mismatch": gold_bad}))

    errors = [c["name"] for c in checks if not c["ok"] and c["severity"] == "error"]
    warns = [c["name"] for c in checks if not c["ok"] and c["severity"] == "warn"]
    return {
        "ok": not errors,
        "dataset": profile.NAME,
        "summary": {
            "records": len(docs),
            "selected": got_selected,
            "tiers": dict(sorted(tier_counts.items())),
            "chunks": len(chunks),
            "errors": errors,
            "warnings": warns,
        },
        "checks": checks,
    }
