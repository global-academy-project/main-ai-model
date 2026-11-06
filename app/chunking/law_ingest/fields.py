"""S5 부가 필드: 조각 → 임베딩 직전 청크 레코드.

레코드 하나가 `/ingest/chunks`의 청크 하나가 된다.
  text         → payload.text (검색 결과·인용에 보이는 원문, 머리말 없음)
  embed_text   → 요청의 chunk.text (임베딩 입력, 머리말을 붙일 수 있음)
id(kb_source_id, kb_chunk_id, point_id)는 비워 둔다(뒤로 미룸). 대신 `natural_key`로
나중에 같은 조각을 다시 찾을 수 있다.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from . import config
from .config import BuildSettings
from .models import Article, ParsedLaw, Piece
from .render import article_lines
from .tags import tags_for


def make_law_key(name: str) -> str:
    return re.sub(r"\s+", "", name)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _year(law: ParsedLaw, basis: str) -> int | None:
    meta = law.meta
    date = meta.promulgation_date if basis == "promulgation" else meta.effective_date
    date = date or meta.effective_date or meta.promulgation_date
    return int(date[:4]) if date else None


def _path_labels(law: ParsedLaw, article: Article) -> list[str]:
    return [law.headings[i].label for i in article.path_idx if law.headings[i].label]


def _hierarchy(law: ParsedLaw, article: Article) -> dict[str, str]:
    """단계별 제목을 따로 담는다: {"편": "제2편 안전기준", …, "속": "제1속 굴착면의 기울기 등"}. 없는 단계는 키가 없다."""
    return {law.headings[i].level: law.headings[i].label for i in article.path_idx if law.headings[i].label}


def _locator(article: Article, piece: Piece) -> str:
    if piece.span:
        return f"{article.label} {piece.span}"
    return article.label


def build_records(law: ParsedLaw, pieces: list[Piece], settings: BuildSettings | None = None) -> list[dict[str, Any]]:
    settings = settings or BuildSettings()
    law_key = settings.law_key or make_law_key(law.meta.law_name)
    file_path = settings.file_path or f"raw/{law.meta.source_file}"
    by_key = {a.key: a for a in law.articles}
    year = _year(law, settings.year_basis)

    # 같은 조 안에서 locator가 겹치면 (k/n)을 붙여 유일하게 만든다.
    seen: dict[tuple[str, str], int] = {}
    for p in pieces:
        loc = _locator(by_key[p.article_key], p)
        seen[(p.article_key, loc)] = seen.get((p.article_key, loc), 0) + 1

    records: list[dict[str, Any]] = []
    for p in pieces:
        art = by_key[p.article_key]
        locator = _locator(art, p)
        if seen[(p.article_key, locator)] > 1:
            locator = f"{locator} ({p.part + 1}/{p.parts_total})"
        text = "\n".join(p.lines)
        path_labels = _path_labels(law, art)
        embed_text = text
        if settings.add_header:
            header = " > ".join([law.meta.law_name, *path_labels])
            embed_text = f"{header}\n{text}"

        work_type: list[str] = []
        hazard: list[str] = []
        if settings.tags_enabled:
            work_type, hazard = tags_for(path_labels, art.key)

        records.append(
            {
                "natural_key": {"law_key": law_key, "article_key": art.key, "part": p.part},
                "article_label": art.label,
                "article_title": art.title,
                "parts_total": p.parts_total,
                "repeated_lead_lines": p.repeated_lead_lines,
                "oversize": p.oversize,
                "path": path_labels,
                "hierarchy": _hierarchy(law, art),
                "text": text,
                "embed_text": embed_text,
                "title": law.meta.law_name,
                "locator": locator,
                "source_type": config.SOURCE_TYPE,
                "collection": config.COLLECTION,
                "site_id": config.SITE_ID,
                "flagged": config.FLAGGED,
                "year": year,
                "hazard": hazard,
                "work_type": work_type,
                "cas": [],
                "url": settings.url,
                "file_path": file_path,
                "article_sha256": _sha("\n".join(article_lines(art))),
                "law_meta": {
                    "effective_date": law.meta.effective_date,
                    "promulgation_no": law.meta.promulgation_no,
                    "promulgation_date": law.meta.promulgation_date,
                    "amendment_type": law.meta.amendment_type,
                    "source_sha256": law.meta.source_sha256,
                },
                # 뒤로 미룬 id. 장부(agent_kb)와 point_id 규칙이 정해지면 채운다.
                "kb_source_id": None,
                "kb_chunk_id": None,
                "point_id": None,
            }
        )
    return records


def to_ingest_item(record: dict[str, Any]) -> dict[str, Any]:
    """레코드 → POST /ingest/chunks 의 chunks[] 한 항목. id가 비어 있으면 거부한다."""
    missing = [k for k in ("kb_source_id", "kb_chunk_id", "point_id") if record.get(k) is None]
    if missing:
        raise ValueError(f"id가 아직 없습니다: {', '.join(missing)}")
    payload = {
        "kb_source_id": record["kb_source_id"],
        "kb_chunk_id": record["kb_chunk_id"],
        "source_type": record["source_type"],
        "site_id": record["site_id"],
        "flagged": record["flagged"],
        "hazard": record["hazard"],
        "work_type": record["work_type"],
        "cas": record["cas"],
        "year": record["year"],
        "title": record["title"],
        "locator": record["locator"],
        "url": record["url"],
        "text": record["text"],
    }
    return {"point_id": record["point_id"], "text": record["embed_text"], "payload": payload}
