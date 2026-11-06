"""S2 구조 파싱: SourceDoc(문단 목록) → ParsedLaw(편·장·절·관 / 조 / 항·호·목 / 부칙).

한글 문서의 스타일은 법령 구조를 담고 있지 않아(모든 문단이 같은 스타일), 글자 모양으로 구분한다.
  편·장·절·관·속 제목  제3편 …, 제10장 …, 제1절 …, 제1관 …, 제1속 …
  조 시작           앞 공백 없이 제N조 / 제N조의M 로 시작하고 (제목) 또는 '삭제'가 이어짐
  항 / 호 / 목       ① / 1. / 가.
  별행 주기          [본조신설 …], [제목개정 …], [제N조에서 이동 …] 처럼 줄 전체가 [ ]
  부칙              '부칙'으로 시작하는 줄부터 끝까지

이 단계는 글자를 고치지 않는다(개정 표시 제거·공백 정리는 clean 단계).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from . import config
from .models import NODE_RANK, Article, Heading, Node, ParsedLaw, SourceDoc

# 제목 단계는 config.HEADING_LEVELS(위 → 아래). 속: 관 아래 하위 제목(예: "제1속 굴착면의 기울기 등")
_KNOWN_UNITS = frozenset("조항호목")  # 제목이 아닌 번호 단위
# 앞에 공백이 있고 "제N<한 글자> 제목"처럼 보이지만 단계 목록에 없는 줄 → 새 제목 단계일 수 있어 경고한다.
_UNKNOWN_HEADING_RE = re.compile(r"^\s{3,}제\d+(?:의\d+)?([가-힣])\s+\S")


@lru_cache(maxsize=None)
def _heading_re(levels: tuple[str, ...]) -> re.Pattern[str]:
    units = "|".join(re.escape(x) for x in levels)
    return re.compile(rf"^\s*제(\d+)(?:의(\d+))?({units})(?:\s+(.*))?$")


_ARTICLE_RE = re.compile(r"^제(\d+)조(?:의(\d+))?(?=[\s(<]|$)")
_NOTE_RE = re.compile(r"^\s*\[[^\[\]]{1,200}\]\s*$")
_ADDENDA_RE = re.compile(r"^\s*부칙(?:[\s<(]|$)")
_TAG_RE = re.compile(r"<[^<>]*>")

_HANG_RE = re.compile(r"^\s*([①-⑳㉑-㉟㊱-㊿])\s*(.*)$")
_HO_RE = re.compile(r"^\s*(\d+(?:의\d+)?\.)(?:\s+(.*))?$")
_MOK_RE = re.compile(r"^\s*([가-힣]\.)(?:\s+(.*))?$")


class ParseError(ValueError):
    """구조를 읽을 수 없는 문서."""


def _balanced_paren(text: str) -> tuple[str, str] | None:
    """text가 '('로 시작할 때 짝이 맞는 ')'까지를 (안쪽, 나머지)로 나눈다."""
    if not text.startswith("("):
        return None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:i], text[i + 1 :]
    return None


def _match_clause(line: str) -> tuple[str, str, str] | None:
    """항/호/목 줄이면 (kind, marker, text)."""
    m = _HANG_RE.match(line)
    if m:
        return "항", m.group(1), m.group(2) or ""
    m = _HO_RE.match(line)
    if m:
        return "호", m.group(1), m.group(2) or ""
    m = _MOK_RE.match(line)
    if m:
        return "목", m.group(1), m.group(2) or ""
    return None


def _is_article_deleted(rest: str) -> bool:
    return _TAG_RE.sub("", rest).strip() == "삭제"


def _parse_article_line(line: str) -> dict[str, Any] | None:
    """조 시작 줄이면 번호·제목·나머지를 돌려준다. 아니면 None."""
    m = _ARTICLE_RE.match(line)
    if not m:
        return None
    number, branch = m.group(1), m.group(2)
    rest = line[m.end() :]
    title: str | None = None
    stripped = rest.lstrip()
    if stripped.startswith("("):
        split = _balanced_paren(stripped)
        if split is None:
            return None
        title, rest = split
    rest = rest.strip()
    deleted = _is_article_deleted(rest)
    if title is None and not deleted:
        return None  # 제N조로 시작하지만 조 시작이 아닌 줄(예: "제5조에 따라 …")
    key = number + (f"-{branch}" if branch else "")
    label = f"제{number}조" + (f"의{branch}" if branch else "")
    return {"key": key, "label": label, "title": title, "rest": rest, "deleted": deleted}


def _is_heading(line: str, levels: tuple[str, ...]) -> re.Match[str] | None:
    # "제2장에 따라 …" 같은 본문은 단위 글자 바로 뒤에 공백이 없어 이 패턴에서 이미 걸러진다.
    return _heading_re(levels).match(line)


class _Builder:
    def __init__(self, levels: tuple[str, ...]) -> None:
        self.level_order = {lv: i for i, lv in enumerate(levels)}
        self.headings: list[Heading] = []
        self.articles: list[Article] = []
        self.path: list[int] = []  # headings 안의 위치
        self.cur: Article | None = None
        self.stack: list[Node] = []
        self.last_index = 0
        self.orphans: list[str] = []
        self.warnings: list[dict[str, Any]] = []

    def warn(self, code: str, index: int, text: str) -> None:
        self.warnings.append({"code": code, "index": index, "text": text.strip()[:120]})

    # -- 제목 --
    def add_heading(self, m: re.Match[str], raw: str, index: int) -> None:
        self.cur = None
        self.stack = []
        number = m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")
        level = m.group(3)
        heading = Heading(level=level, number=number, title=(m.group(4) or "").strip(), raw=raw.strip(), src=index)
        self.headings.append(heading)
        order = self.level_order[level]
        self.path = [i for i in self.path if self.level_order[self.headings[i].level] < order]
        self.path.append(len(self.headings) - 1)

    # -- 조 --
    def start_article(self, info: dict[str, Any], raw: str, index: int) -> None:
        article = Article(
            order=len(self.articles),
            key=info["key"],
            label=info["label"],
            title=info["title"],
            status="DELETED" if info["deleted"] else "ACTIVE",
            path_idx=list(self.path),
            inline_first=False,
            nodes=[],
            notes=[],
            amend_tags=[],
            raw_lines=[raw],
            src_start=index,
            src_end=index,
        )
        self.articles.append(article)
        self.cur = article
        self.stack = []
        rest = info["rest"]
        if info["deleted"] or not rest:
            return
        article.inline_first = True
        clause = _match_clause(rest)
        if clause is not None and clause[0] == "항":
            self.attach(Node("항", clause[1], clause[2].strip()))
        else:
            self.attach(Node("lead", "", rest))

    def attach(self, node: Node) -> None:
        assert self.cur is not None
        rank = NODE_RANK[node.kind]
        while self.stack and NODE_RANK[self.stack[-1].kind] >= rank:
            self.stack.pop()
        if self.stack:
            self.stack[-1].children.append(node)
        else:
            self.cur.nodes.append(node)
        self.stack.append(node)

    def add_body_line(self, line: str, index: int) -> None:
        if self.cur is None:
            self.orphans.append(line)
            self.warn("LINE_OUTSIDE_ARTICLE", index, line)
            return
        art = self.cur
        art.raw_lines.append(line)
        art.src_end = index
        if art.status == "DELETED":
            self.warn("TEXT_AFTER_DELETED_ARTICLE", index, line)
            return
        if _NOTE_RE.match(line):
            art.notes.append(line.strip())
            return
        clause = _match_clause(line)
        if clause is not None:
            self.attach(Node(clause[0], clause[1], clause[2].strip()))
            return
        # 번호 없는 이어지는 줄
        if self.stack:
            self.stack[-1].lines.append(line.strip())
        else:
            art.nodes.append(Node("lead", "", line.strip()))
            self.stack.append(art.nodes[-1])
            self.warn("LEAD_ON_NEXT_LINE", index, line)


def parse_source(doc: SourceDoc, levels: tuple[str, ...] | None = None) -> ParsedLaw:
    levels = tuple(levels or config.HEADING_LEVELS)
    if len(set(levels)) != len(levels) or any(len(x) != 1 for x in levels):
        raise ParseError("제목 단계는 한 글자씩, 중복 없이 지정해야 합니다.")
    lines: list[tuple[int, str]] = []
    for p in doc.paragraphs:
        for ln in p.text.split("\n"):
            lines.append((p.index, ln))

    body_pos = None
    for pos, (_, text) in enumerate(lines):
        if _is_heading(text, levels) or _parse_article_line(text):
            body_pos = pos
            break
    if body_pos is None:
        raise ParseError("편·장·조 제목을 찾지 못했습니다. 법령 문서가 맞는지 확인하세요.")

    prelude = [t for _, t in lines[:body_pos] if t.strip()]
    b = _Builder(levels)
    addenda: list[str] = []
    addenda_src: int | None = None
    body_end = lines[body_pos][0]

    for pos in range(body_pos, len(lines)):
        index, text = lines[pos]
        if addenda_src is not None:
            addenda.append(text)
            continue
        body_end = index
        if not text.strip():
            if b.cur is not None:
                b.cur.src_end = index
            continue
        if b.articles and _ADDENDA_RE.match(text):
            addenda_src = index
            addenda.append(text)
            body_end = index - 1
            continue
        h = _is_heading(text, levels)
        if h:
            b.add_heading(h, text, index)
            continue
        info = _parse_article_line(text)
        if info is not None and not text[:1].isspace():
            b.start_article(info, text, index)
            continue
        if info is None and _ARTICLE_RE.match(text):
            b.warn("ARTICLE_LIKE_LINE_NOT_STARTED", index, text)
        um = _UNKNOWN_HEADING_RE.match(text)
        if um and um.group(1) not in _KNOWN_UNITS and um.group(1) not in levels:
            b.warn("UNKNOWN_HEADING_LEVEL", index, text)
        b.add_body_line(text, index)

    return ParsedLaw(
        meta=doc.meta,
        stage="parsed",
        headings=b.headings,
        articles=b.articles,
        prelude=prelude,
        addenda=addenda,
        body_src_start=lines[body_pos][0],
        body_src_end=body_end,
        addenda_src_start=addenda_src,
        orphans=b.orphans,
        warnings=b.warnings,
    )
