"""S3 정리: 개정 표시·삭제·공백을 본문에서 분리한다. 글자 보존을 장부로 남긴다.

하는 일
  - 문장 끝 `<개정 2017. 3. 3.>`, `삭제<2024. 6. 28.>`의 꺾쇠 표시를 떼어 `amend_tags`로 보관
  - 번호만 남은 삭제 호/항(`3. 삭제`)은 `deleted`로 표시(번호는 당기지 않음)
  - 공백(연속 공백·nbsp 등)을 한 칸으로 정리
  - 별행 주기(`[본조신설 …]`)는 parse가 이미 `notes`로 분리했다. 여기서는 글자 수만 센다.
하지 않는 일
  - 문장 부호·한자·ㆍ·「」·원문자(①)는 건드리지 않는다.

조마다 `ledger`에 공백 아닌 글자 수를 적는다.
  raw = kept + tags + notes + deleted_nodes + deleted_article
검증(validate)이 이 식을 다시 계산한다.
"""

from __future__ import annotations

import copy
import re

from .models import Article, Heading, Node, ParsedLaw, nonspace
from .render import article_blocks, article_lines

# 꺾쇠 안에 날짜(2017. 3. 3.)가 있는 것만 개정 표시로 본다. <표 2> 같은 본문은 그대로 둔다.
_TAG_RE = re.compile(r"\s*(<[^<>]*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.[^<>]*>)")


def _strip_tags(text: str) -> tuple[str, list[str], int]:
    tags: list[str] = []

    def take(m: re.Match[str]) -> str:
        tags.append(m.group(1))
        return " "

    out = _TAG_RE.sub(take, text)
    return out, tags, sum(nonspace(t) for t in tags)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _clean_node(node: Node, acc: dict[str, int], tags: list[str]) -> None:
    node.text, t, n = _strip_tags(node.text)
    tags.extend(t)
    acc["tags"] += n
    node.text = _norm(node.text)
    new_lines = []
    for ln in node.lines:
        ln, t, n = _strip_tags(ln)
        tags.extend(t)
        acc["tags"] += n
        ln = _norm(ln)
        if ln:
            new_lines.append(ln)
    node.lines = new_lines
    for child in node.children:
        _clean_node(child, acc, tags)
    if node.text == "삭제" and not node.children and not node.lines:
        node.deleted = True
        acc["deleted_nodes"] += nonspace(node.marker) + nonspace(node.text)


def _clean_heading(h: Heading) -> None:
    title, tags, _ = _strip_tags(h.title)
    title = _norm(title)
    h.deleted = title == "삭제"
    h.title = title
    h.amend_tags = tags
    h.label = f"제{h.number}{h.level}" + (f" {title}" if title else "")


def _clean_article(a: Article) -> None:
    acc = {"tags": 0, "notes": 0, "deleted_nodes": 0, "deleted_article": 0}
    raw = sum(nonspace(ln) for ln in a.raw_lines)
    tags: list[str] = []

    if a.title is not None:
        a.title, t, n = _strip_tags(a.title)
        tags.extend(t)
        acc["tags"] += n
        a.title = _norm(a.title)

    if a.status == "DELETED":
        # 조 전체가 삭제. 제목 줄의 표시까지 모두 '삭제된 조'로 센다.
        for ln in a.raw_lines:
            tags.extend(m.group(1) for m in _TAG_RE.finditer(ln))
        a.nodes = []
        a.notes = [_norm(x) for x in a.notes]
        a.amend_tags = tags
        a.ledger = {"raw": raw, "kept": 0, "tags": 0, "notes": 0, "deleted_nodes": 0, "deleted_article": raw}
        return

    for node in a.nodes:
        _clean_node(node, acc, tags)
    a.notes = [_norm(x) for x in a.notes]
    acc["notes"] = sum(nonspace(x) for x in a.notes)
    a.amend_tags = tags
    if not article_blocks(a):
        a.status = "EMPTY"
    kept = sum(nonspace(ln) for ln in article_lines(a)) if a.status == "ACTIVE" else 0
    if a.status == "EMPTY":
        # 본문이 모두 삭제된 조: 제목 줄만 남으므로 제목 줄 글자도 삭제분으로 센다.
        acc["deleted_article"] = raw - acc["tags"] - acc["notes"] - acc["deleted_nodes"]
    a.ledger = {"raw": raw, "kept": kept, **acc}


def clean_law(law: ParsedLaw) -> ParsedLaw:
    out = copy.deepcopy(law)
    out.stage = "cleaned"
    for h in out.headings:
        _clean_heading(h)
    for a in out.articles:
        _clean_article(a)
        if a.status == "EMPTY":
            out.warnings.append({"code": "EMPTY_ARTICLE", "index": a.src_start, "text": a.label})
    return out
