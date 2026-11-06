"""S7 갱신 계획: 이전에 만든 청크와 이번에 만든 청크를 비교해 무엇을 올리고 무엇을 지울지 계산한다.

법령이 개정되면 hwpx를 다시 받아 같은 파이프라인을 돌린 뒤, 이전 작업 폴더와 비교한다.
  add        이번에만 있는 청크            → 새로 올림
  update     양쪽에 있고 내용이 달라진 청크 → 같은 point_id로 덮어씀(upsert)
  remove     이전에만 있는 청크            → Qdrant에서 지움(삭제된 조, 줄어든 조각, 사라진 조)
  unchanged  내용이 같은 청크              → 건드리지 않음

청크의 정체성은 natural_key(law_key, article_key, part)다. id(kb_source_id, point_id)는 아직 없으므로
이 단계는 id를 만들지 않고, id가 정해지면 같은 natural_key로 찾아 쓴다.
내용 비교는 Qdrant에 실제로 들어가는 값(임베딩 입력, 인용 원문, 제목, 위치, 연도, 태그, url)만 본다.
작업 폴더 이름·원본 파일 이름(file_path)이 바뀐 것만으로는 갱신하지 않는다.

안전장치(--allow-* 로만 해제)
  - 법령 식별자(law_key)가 다르면 거부
  - 이전 판보다 시행일이 앞선 파일이면 거부(옛 판으로 덮어쓰기 방지)
  - 지울 청크가 이전의 10%를 넘으면 거부(파싱 실패로 조가 통째로 사라지는 사고 방지)
  - 이번 검증(06_validation.json)이 통과하지 않았으면 거부
"""

from __future__ import annotations

import difflib
import hashlib
import json
from typing import Any

# 바뀌면 Qdrant 쪽 값도 바뀌어야 하는 필드
_CONTENT_FIELDS = (
    "embed_text",
    "text",
    "title",
    "locator",
    "year",
    "hazard",
    "work_type",
    "cas",
    "site_id",
    "flagged",
    "source_type",
    "collection",
    "url",
)

MAX_REMOVE_RATIO = 0.10
_DIFF_PREVIEW_LINES = 30


def key_of(record: dict[str, Any]) -> tuple[str, str, int]:
    nk = record["natural_key"]
    return nk["law_key"], nk["article_key"], int(nk["part"])


def key_str(key: tuple[str, str, int]) -> str:
    return f"{key[0]}:{key[1]}:{key[2]}"


def content_digest(record: dict[str, Any]) -> str:
    payload = {name: record.get(name) for name in _CONTENT_FIELDS}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "natural_key": record["natural_key"],
        "article_label": record["article_label"],
        "locator": record["locator"],
    }


def _preview(old: str, new: str) -> list[str]:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "이전", "이번", lineterm="", n=0))
    if len(lines) > _DIFF_PREVIEW_LINES:
        lines = lines[:_DIFF_PREVIEW_LINES] + [f"… (+{len(lines) - _DIFF_PREVIEW_LINES}줄 생략)"]
    return lines


def _law_meta(records: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(records[0]["law_meta"]) if records else {}


def plan_update(
    prev: list[dict[str, Any]],
    new: list[dict[str, Any]],
    new_articles: dict[str, str] | None = None,
    *,
    allow_older: bool = False,
    allow_mass_remove: bool = False,
) -> dict[str, Any]:
    """prev·new = 05_chunks.jsonl 레코드 목록. new_articles = {article_key: status}(이번 03_cleaned의 조 상태)."""
    if not new:
        raise ValueError("이번 청크가 비어 있습니다.")
    if not prev:
        raise ValueError("이전 청크가 비어 있습니다. 처음 적재라면 비교 없이 전체를 올리면 됩니다.")
    new_articles = new_articles or {}

    prev_laws = {key_of(r)[0] for r in prev}
    new_laws = {key_of(r)[0] for r in new}
    if prev_laws != new_laws:
        raise ValueError(f"법령 식별자가 다릅니다: 이전 {sorted(prev_laws)} / 이번 {sorted(new_laws)}")

    pm, nm = _law_meta(prev), _law_meta(new)
    p_eff, n_eff = pm.get("effective_date"), nm.get("effective_date")
    if p_eff and n_eff and n_eff < p_eff and not allow_older:
        raise ValueError(f"이번 파일의 시행일({n_eff})이 이전({p_eff})보다 앞섭니다. 옛 판으로 덮어쓰려면 --allow-older.")

    pmap = {key_of(r): r for r in prev}
    nmap = {key_of(r): r for r in new}
    if len(pmap) != len(prev) or len(nmap) != len(new):
        raise ValueError("natural_key가 중복된 청크가 있습니다.")

    add, update, unchanged, remove = [], [], [], []
    for k in sorted(nmap, key=lambda x: (x[0], _sort_key(x[1]), x[2])):
        r = nmap[k]
        if k not in pmap:
            add.append(_entry(r))
        elif content_digest(pmap[k]) != content_digest(r):
            e = _entry(r)
            e["old_locator"] = pmap[k]["locator"]
            e["diff"] = _preview(pmap[k]["text"], r["text"])
            if not e["diff"] and pmap[k]["embed_text"] != r["embed_text"]:
                e["diff"] = ["(본문은 같고 임베딩 머리글·태그·연도 등 부가 값이 달라짐)"]
            update.append(e)
        else:
            unchanged.append(_entry(r))

    new_keys_by_article = {k[1] for k in nmap}
    for k in sorted(pmap, key=lambda x: (x[0], _sort_key(x[1]), x[2])):
        if k in nmap:
            continue
        e = _entry(pmap[k])
        status = new_articles.get(k[1])
        if k[1] in new_keys_by_article:
            e["reason"] = "PART_DROPPED"  # 조는 남았지만 조각 수가 줄었다
        elif status == "DELETED":
            e["reason"] = "ARTICLE_DELETED"  # 법에서 삭제된 조
        elif status is not None:
            e["reason"] = f"ARTICLE_{status}"
        else:
            e["reason"] = "ARTICLE_GONE"  # 이번 판에서 조 자체를 찾지 못함(조 이동·번호 변경이거나 파싱 문제)
        remove.append(e)

    gone = [e for e in remove if e["reason"] == "ARTICLE_GONE"]
    ratio = len(remove) / len(prev)
    if ratio > MAX_REMOVE_RATIO and not allow_mass_remove:
        raise ValueError(
            f"지울 청크가 이전의 {ratio:.0%}({len(remove)}/{len(prev)})입니다. "
            f"파싱이 실패한 것이 아닌지 확인하고, 맞다면 --allow-mass-remove."
        )

    reasons: dict[str, int] = {}
    for e in remove:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    return {
        "law": new[0]["title"],
        "prev_law_meta": pm,
        "new_law_meta": nm,
        "summary": {
            "prev_chunks": len(prev),
            "new_chunks": len(new),
            "add": len(add),
            "update": len(update),
            "remove": len(remove),
            "unchanged": len(unchanged),
            "remove_reasons": reasons,
            "same_source_file": bool(pm.get("source_sha256")) and pm.get("source_sha256") == nm.get("source_sha256"),
            "needs_review": [f"{len(gone)}개 청크의 조가 이번 판에서 사라짐(ARTICLE_GONE) — 파싱 누락인지 확인"] if gone else [],
        },
        "add": add,
        "update": update,
        "remove": remove,
    }


def _sort_key(article_key: str) -> tuple[int, int]:
    head, _, branch = article_key.partition("-")
    return int(head), int(branch or 0)
