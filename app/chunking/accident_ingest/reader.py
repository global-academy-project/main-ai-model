"""S1 읽기: 받아 둔 API 원본 페이지(JSON) → 레코드 목록. 값은 건드리지 않는다.

입력은 폴더 하나(페이지마다 파일 하나). 파일 이름은 상관없고 내용의 header/body로 판단한다.
- header.resultCode != "00" 이면 거부한다(HTTP 200이어도 오류일 수 있다).
- body.items.item 이 배열이면 그대로, 객체면 한 건짜리 배열, 비었거나 문자열이면 0건.
- 업종(cateSeNm)별로 페이지가 1..N 연속이고 건수 합이 totalCount 와 같은지 본다(complete).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import sha256_text


class ReaderError(Exception):
    pass


CATE_ORDER = ("건설업", "전업종")


def _items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items = body.get("items")
    if not isinstance(items, dict):
        return []
    item = items.get("item")
    if isinstance(item, list):
        return [x for x in item if isinstance(x, dict)]
    if isinstance(item, dict):
        return [item]
    return []


def read_pages(raw_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = Path(raw_dir)
    files = sorted(p for p in raw_dir.glob("*.json") if p.name not in {"fetch_manifest.json"})
    if not files:
        raise ReaderError(f"원본 페이지(JSON)가 없습니다: {raw_dir}")

    pages: list[dict[str, Any]] = []
    for path in files:
        data_bytes = path.read_bytes()
        try:
            data = json.loads(data_bytes.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReaderError(f"{path.name}: JSON이 아닙니다 ({exc})") from exc
        header = data.get("header") if isinstance(data, dict) else None
        body = data.get("body") if isinstance(data, dict) else None
        if not isinstance(header, dict) or not isinstance(body, dict):
            raise ReaderError(f"{path.name}: header/body 모양이 아닙니다.")
        code = str(header.get("resultCode"))
        if code != "00":
            raise ReaderError(f"{path.name}: resultCode={code} ({header.get('resultMsg')}). 정상 응답만 읽습니다.")
        items = _items(body)
        cates = sorted({str(i.get("cateSeNm")) for i in items})
        if len(cates) > 1:
            raise ReaderError(f"{path.name}: 한 페이지에 업종이 섞여 있습니다: {cates}")
        pages.append(
            {
                "file": path.name,
                "sha256": sha256_text(data_bytes.decode("utf-8-sig")),
                "cate": cates[0] if cates else None,
                "result_code": code,
                "page_no": int(body.get("pageNo", 0)),
                "num_of_rows": int(body.get("numOfRows", 0)),
                "total_count": int(body.get("totalCount", -1)),
                "count": len(items),
                "_items": items,
            }
        )

    pages.sort(key=lambda p: (CATE_ORDER.index(p["cate"]) if p["cate"] in CATE_ORDER else 99, str(p["cate"]), p["page_no"]))
    seen: set[tuple[str | None, int]] = set()
    for p in pages:
        k = (p["cate"], p["page_no"])
        if k in seen:
            raise ReaderError(f"같은 업종·페이지가 두 번 있습니다: {k} ({p['file']})")
        seen.add(k)

    records: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for p in pages:
        for idx, item in enumerate(p.pop("_items")):
            rec = dict(item)
            rec["_src"] = {"file": p["file"], "page": p["page_no"], "index": idx}
            records.append(rec)
        g = groups.setdefault(str(p["cate"]), {"total_counts": set(), "pages": [], "items": 0})
        g["total_counts"].add(p["total_count"])
        g["pages"].append(p["page_no"])
        g["items"] += p["count"]

    out_groups: dict[str, Any] = {}
    for cate, g in groups.items():
        totals = sorted(g["total_counts"])
        contiguous = g["pages"] == list(range(1, len(g["pages"]) + 1))
        complete = len(totals) == 1 and contiguous and g["items"] == totals[0]
        out_groups[cate] = {"total_count": totals[0] if len(totals) == 1 else totals, "pages": g["pages"], "items": g["items"], "complete": complete}
    manifest = {
        "pages": pages,
        "groups": out_groups,
        "records": len(records),
        "complete": bool(out_groups) and all(g["complete"] for g in out_groups.values()),
    }
    return records, manifest
