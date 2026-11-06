"""S6 검증: 청크 파일을 확정해도 되는지 확인한다. 하나라도 error면 확정하지 않는다.

확인 항목
  articles_unique        조 번호(key) 중복 없음
  toc_match              (목차 파일이 있으면) 조 목록·순서가 목차와 같음. 목차에만 있는 조는 '이동' 안내가 있어야 함
  article_ledger         조마다 raw = kept + tags + notes + deleted_nodes + deleted_article (공백 아닌 글자 수)
  body_conservation      원본 본문 글자 수 = 제목 + 조 + 어느 조에도 못 든 줄
  kept_matches_chunks    정리된 조 글자 수 = 청크로 나간 글자 수(되풀이한 줄 제외)
  split_reconstruction   조각을 이으면 원래 조와 같음
  chunk_coverage         ACTIVE 조는 청크가 있고, 그 밖의 조는 청크가 없음
  chunk_fields           빈 본문·한도 초과·필수 필드 누락 없음
  natural_keys_unique    (law_key, article_key, part) 중복 없음, 조 안 locator 중복 없음
  no_noise               머리글·바닥글 잡음이 청크에 없음
  heading_leak           제목 줄(제N장 … / 제N속 …)이 조 본문에 섞이지 않음
  unknown_heading_level  목록에 없는 제목 단계(제N<새 단계> …)로 보이는 줄이 없음
  orphan_lines           어느 조에도 속하지 못한 줄(경고)
  addenda_excluded       부칙 글이 청크에 없음
  golden                 기준 조문에 원문 값이 들어 있음 (알려진 법령일 때)
"""

from __future__ import annotations

import re
from typing import Any

from . import config
from .golden import GOLDEN
from .models import ParsedLaw, SourceDoc, nonspace
from .render import article_lines

_TOC_ARTICLE_RE = re.compile(r"^\s*(제\d+조(?:의\d+)?)")
# 제목 줄이 조 본문에 섞여 들어왔는지 찾는 패턴(편·장·절·관·속). 조(條)는 제외.
def _heading_leak_re(parsed: ParsedLaw) -> re.Pattern[str]:
    levels = set(config.HEADING_LEVELS) | {h.level for h in parsed.headings}
    return re.compile(rf"^\s*제\d+(?:의\d+)?({'|'.join(sorted(levels))})\s")


def _check(name: str, ok: bool, detail: Any = None, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "detail": detail}


def _chunks_by_article(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        out.setdefault(r["natural_key"]["article_key"], []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: r["natural_key"]["part"])
    return out


def _squash(text: str) -> str:
    return "".join(text.split())


def validate_all(
    source: SourceDoc | None,
    parsed: ParsedLaw,
    cleaned: ParsedLaw,
    records: list[dict[str, Any]],
    toc_lines: list[str] | None = None,
    expect_articles: int | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    arts = cleaned.articles
    by_status: dict[str, int] = {}
    for a in arts:
        by_status[a.status] = by_status.get(a.status, 0) + 1

    # 1. 조 번호 중복
    keys = [a.key for a in arts]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    checks.append(_check("articles_unique", not dups, {"duplicates": dups, "count": len(arts), "by_status": by_status}))
    if expect_articles is not None:
        checks.append(_check("articles_expected", len(arts) == expect_articles, {"expected": expect_articles, "actual": len(arts)}))

    # 2. 목차 대조
    if toc_lines is not None:
        toc = [m.group(1) for ln in toc_lines if (m := _TOC_ARTICLE_RE.match(ln))]
        have = [a.label for a in arts]
        have_set = set(have)
        toc_set = set(toc)
        notes_text = " ".join(n for a in arts for n in a.notes)
        only_toc = [t for t in toc if t not in have_set]
        moved = [t for t in only_toc if f"{t}에서 이동" in notes_text]
        unexplained = [t for t in only_toc if t not in moved]
        only_have = [h for h in have if h not in toc_set]
        order_ok = [h for h in have if h in toc_set] == [t for t in toc if t in have_set]
        checks.append(
            _check(
                "toc_match",
                not unexplained and not only_have and order_ok,
                {
                    "toc_count": len(toc),
                    "parsed_count": len(have),
                    "moved_explained": moved,
                    "missing_unexplained": unexplained,
                    "extra_in_parsed": only_have,
                    "order_same": order_ok,
                },
            )
        )

    # 3. 조별 글자 보존 장부
    bad = []
    for a in cleaned.articles:
        lg = a.ledger
        kept = sum(nonspace(ln) for ln in article_lines(a)) if a.status == "ACTIVE" else 0
        total = kept + lg.get("tags", 0) + lg.get("notes", 0) + lg.get("deleted_nodes", 0) + lg.get("deleted_article", 0)
        raw = sum(nonspace(ln) for ln in parsed.articles[a.order].raw_lines)
        if lg.get("raw") != raw or total != raw:
            bad.append({"article": a.label, "raw": raw, "ledger_raw": lg.get("raw"), "sum": total})
    checks.append(_check("article_ledger", not bad, {"mismatch": bad[:10], "count": len(bad)}))

    # 4. 원본 본문 보존
    if source is not None:
        body = sum(
            nonspace(p.text)
            for p in source.paragraphs
            if parsed.body_src_start <= p.index <= parsed.body_src_end
        )
        headings = sum(nonspace(h.raw) for h in parsed.headings)
        articles = sum(sum(nonspace(ln) for ln in a.raw_lines) for a in parsed.articles)
        orphans = sum(nonspace(x) for x in parsed.orphans)
        checks.append(
            _check(
                "body_conservation",
                body == headings + articles + orphans,
                {"source_body": body, "headings": headings, "articles": articles, "orphans": orphans},
            )
        )

    # 5. 청크로 나간 글자 = 정리된 조 글자 / 조각을 이으면 원래 조
    by_article = _chunks_by_article(records)
    recon_bad: list[str] = []
    kept_bad: list[dict[str, Any]] = []
    for a in cleaned.articles:
        if a.status != "ACTIVE":
            continue
        full = article_lines(a)
        recs = by_article.get(a.key, [])
        joined: list[str] = []
        for r in recs:
            lines = r["text"].split("\n")
            joined.extend(lines[r["repeated_lead_lines"] :])
        if joined != full:
            recon_bad.append(a.label)
        sent = sum(nonspace(ln) for ln in joined)
        want = sum(nonspace(ln) for ln in full)
        if sent != want:
            kept_bad.append({"article": a.label, "chunks": sent, "expected": want})
    checks.append(_check("split_reconstruction", not recon_bad, {"articles": recon_bad[:10], "count": len(recon_bad)}))
    checks.append(_check("kept_matches_chunks", not kept_bad, {"mismatch": kept_bad[:10], "count": len(kept_bad)}))

    # 6. 청크 대상
    active = {a.key for a in arts if a.status == "ACTIVE"}
    with_chunks = set(by_article)
    checks.append(
        _check(
            "chunk_coverage",
            active == with_chunks,
            {
                "active_articles": len(active),
                "articles_with_chunks": len(with_chunks),
                "without_chunks": sorted(active - with_chunks)[:10],
                "chunks_for_non_active": sorted(with_chunks - active)[:10],
            },
        )
    )

    # 7. 청크 필드
    field_bad: list[dict[str, Any]] = []
    oversize = 0
    long_embed = 0
    for r in records:
        probs = []
        if not r.get("text", "").strip():
            probs.append("EMPTY_TEXT")
        if len(r.get("text", "")) > config.MAX_TEXT_CHARS:
            probs.append("TEXT_TOO_LONG")
        if not r.get("embed_text", "").strip():
            probs.append("EMPTY_EMBED_TEXT")
        for f in ("title", "locator", "source_type", "collection"):
            if not r.get(f):
                probs.append(f"NO_{f.upper()}")
        if not r.get("url") and not r.get("file_path"):
            probs.append("NO_URL_OR_FILE_PATH")
        if r.get("oversize"):
            oversize += 1
        if len(r.get("embed_text", "")) > config.MAX_EMBED_CHARS:
            long_embed += 1
        if probs:
            field_bad.append({"chunk": r["natural_key"], "problems": probs})
    checks.append(_check("chunk_fields", not field_bad, {"bad": field_bad[:10], "count": len(field_bad)}))
    checks.append(_check("chunk_oversize", oversize == 0, {"oversize_chunks": oversize}, severity="warn"))
    checks.append(_check("embed_text_length", long_embed == 0, {"over_limit_chunks": long_embed}, severity="warn"))

    # 8. 자연 키 중복
    nk = [(r["natural_key"]["law_key"], r["natural_key"]["article_key"], r["natural_key"]["part"]) for r in records]
    nk_dups = sorted({k for k in nk if nk.count(k) > 1})
    loc_pairs = [(r["natural_key"]["article_key"], r["locator"]) for r in records]
    loc_dups = sorted({k for k in loc_pairs if loc_pairs.count(k) > 1})
    checks.append(_check("natural_keys_unique", not nk_dups and not loc_dups, {"natural_key_dups": nk_dups[:5], "locator_dups": loc_dups[:5]}))

    # 9. 잡음
    noisy = [
        r["natural_key"]
        for r in records
        if any(s in r["text"] or s in r["embed_text"] for s in config.NOISE_STRINGS)
    ]
    checks.append(_check("no_noise", not noisy, {"chunks": noisy[:10], "strings": list(config.NOISE_STRINGS)}))

    # 9-2. 제목 줄 누수 / 어느 조에도 못 든 줄
    leak_re = _heading_leak_re(parsed)
    leaks = [
        {"chunk": r["natural_key"], "line": ln.strip()[:40]}
        for r in records
        for ln in r["text"].split("\n")
        if leak_re.match(ln)
    ]
    checks.append(_check("heading_leak", not leaks, {"count": len(leaks), "examples": leaks[:5]}))
    unknown = [w for w in cleaned.warnings if w["code"] == "UNKNOWN_HEADING_LEVEL"]
    checks.append(_check("unknown_heading_level", not unknown, {"count": len(unknown), "examples": [w["text"] for w in unknown[:5]]}))
    checks.append(_check("orphan_lines", not parsed.orphans, {"count": len(parsed.orphans), "examples": [o.strip()[:40] for o in parsed.orphans[:5]]}, severity="warn"))

    # 10. 부칙
    add_lines = [_squash(x) for x in parsed.addenda if len(_squash(x)) >= 12]
    leaked = [x[:30] for x in add_lines if any(x in _squash(r["text"]) for r in records)]
    checks.append(
        _check(
            "addenda_excluded",
            not leaked,
            {"addenda_lines": len(parsed.addenda), "found_in_chunks": leaked[:5], "starts_at_paragraph": parsed.addenda_src_start},
        )
    )

    # 11. 기준 조문
    golden = GOLDEN.get(cleaned.meta.law_name)
    if golden is None:
        checks.append(_check("golden", True, "기준값이 없는 법령이라 건너뜀", severity="info"))
    else:
        missing: list[dict[str, Any]] = []
        for label, needles in golden:
            texts = [_squash(r["text"]) for r in records if r["article_label"] == label]
            if not texts:
                missing.append({"article": label, "problem": "청크 없음"})
                continue
            joined = "".join(texts)
            for needle in needles:
                if _squash(needle) not in joined:
                    missing.append({"article": label, "missing": needle})
        checks.append(_check("golden", not missing, {"checked": [g[0] for g in golden], "missing": missing}))

    # 12. 참고: 파싱 경고
    codes: dict[str, int] = {}
    for w in cleaned.warnings:  # cleaned는 parsed 경고를 포함한다
        codes[w["code"]] = codes.get(w["code"], 0) + 1
    checks.append(_check("parse_warnings", not codes, codes, severity="warn"))

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    warns = [c for c in checks if not c["ok"] and c["severity"] == "warn"]
    return {
        "ok": not errors,
        "law": cleaned.meta.law_name,
        "summary": {
            "articles": len(arts),
            "by_status": by_status,
            "chunks": len(records),
            "split_articles": sum(1 for rs in by_article.values() if len(rs) > 1),
            "errors": [c["name"] for c in errors],
            "warnings": [c["name"] for c in warns],
        },
        "checks": checks,
    }
