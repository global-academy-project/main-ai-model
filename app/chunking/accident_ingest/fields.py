"""S5 부가 필드: 조각 → 임베딩 직전 청크 레코드.

레코드 하나가 `/ingest/document` 요청의 청크 하나가 된다(소스 = 사례 하나).
  text         → 청크 text (검색 결과·인용에 보이는 원문. 라벨은 붙고 머리말은 없음)
  embed_text   → chunk.embedding_text (임베딩 입력, 머리말 포함)
id(kb_source_id, kb_chunk_id, point_id)는 서버가 발급·계산한다. 여기서는 만들지 않는다.
`chunk_index = slot * 100 + part` 는 계산식이라 별도 매핑 파일이 없다.
"""

from __future__ import annotations

import re
from datetime import date
from types import ModuleType
from typing import Any

from . import config, flag
from .config import BuildSettings
from .models import one_line, sha256_json
from .tags import tags_for

_NUM_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*")


def strip_num(name: str) -> str:
    """건설업 분류의 번호 접두('1.2.3 ')를 뗀다. 원문 text·장부는 그대로 두고 제목·머리말에서만 쓴다."""
    return _NUM_PREFIX_RE.sub("", name).strip()


def make_title(clean: dict[str, str], year: int | None) -> str:
    base = strip_num(clean["sifSclsfNm"]) or strip_num(clean["sifMclsfNm"]) or strip_num(clean["sifLclsfNm"])
    tail = clean["situation"] or clean["disasterType"]
    parts = [base] if base else []
    if tail and tail != base:
        parts.append(tail)
    title = " – ".join(parts) if parts else "SIF 재해사례"
    if year:
        title += f" ({year})"
    if len(title) > config.MAX_TITLE_CHARS:
        title = title[: config.MAX_TITLE_CHARS - 1] + "…"
    return title


def make_header(clean: dict[str, str], profile: ModuleType) -> str:
    """임베딩 머리말: SIF > 대 > 중 > 소 > 상황(또는 재해종류). 값이 없거나 앞 칸과 같은 칸은 뺀다."""
    chain = [profile.SOURCE_TYPE]
    for v in (clean["sifLclsfNm"], clean["sifMclsfNm"], clean["sifSclsfNm"], clean["situation"] or clean["disasterType"]):
        v = strip_num(one_line(v))
        if v and profile.squash(v) != profile.squash(chain[-1]):
            chain.append(v)
    return " > ".join(chain)


def make_locator(piece: dict[str, Any], label_of: dict[str, str]) -> str:
    if piece["section"] == "case":
        return config.CASE_LOCATOR
    label = label_of[piece["section"]]
    if piece["parts_total"] > 1:
        return f"{label} ({piece['part'] + 1}/{piece['parts_total']})"
    return label


def build_records(
    docs: list[dict[str, Any]],
    pieces: list[dict[str, Any]],
    settings: BuildSettings,
    profile: ModuleType,
) -> list[dict[str, Any]]:
    year_max = settings.year_max or date.today().year
    by_id = {d["external_id"]: d for d in docs}
    label_of = {name: label for name, _, label in config.SECTIONS}
    flagged_ids: dict[str, list[str]] = {}
    for d in docs:
        text = "\n".join(d["clean"][f] for _, f, _ in config.SECTIONS)
        hits = flag.scan(text)
        if hits:
            flagged_ids[d["external_id"]] = hits

    records: list[dict[str, Any]] = []
    for p in pieces:
        d = by_id[p["external_id"]]
        clean = d["clean"]
        tags = tags_for(d, profile, settings, year_max)
        text = "\n".join(p["lines"])
        header = make_header(clean, profile)
        embed_text = f"{header}\n{text}" if settings.add_header else text
        title = make_title(clean, tags["year"])
        chunk_index = config.SLOT[p["section"]] * 100 + p["part"]
        rec: dict[str, Any] = {
            "natural_key": {"dataset": profile.NAME, "record": d["external_id"], "section": p["section"], "part": p["part"]},
            "external_id": d["external_id"],
            "chunk_index": chunk_index,
            "parts_total": p["parts_total"],
            "repeated_lead_lines": p["repeated_lead_lines"],
            "oversize": p["oversize"],
            "text": text,
            "embed_text": embed_text,
            "header": header,
            "title": title,
            "locator": make_locator(p, label_of),
            "source_type": profile.SOURCE_TYPE,
            "collection": profile.COLLECTION,
            "site_id": config.SITE_ID,
            "flagged": d["external_id"] in flagged_ids,
            "flag_hits": flagged_ids.get(d["external_id"], []),
            "year": tags["year"],
            "severity": tags["severity"],
            "accident_type": tags["accident_type"],
            "hazard": tags["hazard"],
            "work_type": tags["work_type"],
            "cas": tags["cas"],
            "ghs": tags["ghs"],
            "url": profile.DATASET_URL,
            "license": profile.LICENSE,
            "publisher": profile.PUBLISHER,
            "content_sha256": d["content_sha256"],
            "tier": d["tier"],
            "cate": clean["cateSeNm"],
            "disaster_type": clean["disasterType"] or None,
            "situation": clean["situation"] or None,
            "kb_source_id": None,
            "kb_chunk_id": None,
            "point_id": None,
        }
        rec["content_digest"] = sha256_json(
            {k: rec[k] for k in ("chunk_index", "text", "embed_text", "title", "locator", "flagged", "year", "severity", "accident_type", "hazard", "work_type", "cas", "ghs")}
        )
        records.append(rec)
    return records


def tag_report(records: list[dict[str, Any]], docs: list[dict[str, Any]]) -> dict[str, Any]:
    """축별 채워진 비율(업종별)과 값 분포. 정답과 대조한 정확도가 아니라 규칙이 몇 건에 붙었는지다."""
    by_cate: dict[str, dict[str, Any]] = {}
    for r in records:
        c = by_cate.setdefault(r["cate"], {"chunks": 0, "filled": {}, "values": {}})
        c["chunks"] += 1
        for axis in ("severity", "accident_type", "hazard", "work_type", "year"):
            v = r[axis]
            filled = bool(v) if isinstance(v, list) else v is not None
            if filled:
                c["filled"][axis] = c["filled"].get(axis, 0) + 1
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    key = f"{axis}={x}" if axis != "year" else "year"
                    if axis != "year":
                        c["values"][key] = c["values"].get(key, 0) + 1
    unmapped: dict[str, dict[str, int]] = {}
    for r in records:
        if r["cate"] == "건설업" and not r["accident_type"]:
            unmapped.setdefault("건설 accident_type 비움(disasterType)", {})
            k = r["disaster_type"] or "(없음)"
            unmapped["건설 accident_type 비움(disasterType)"][k] = unmapped["건설 accident_type 비움(disasterType)"].get(k, 0) + 1
    for c in by_cate.values():
        n = c["chunks"] or 1
        c["filled_ratio"] = {a: round(v / n, 3) for a, v in c["filled"].items()}
    return {"by_cate": by_cate, "unmapped": unmapped}
