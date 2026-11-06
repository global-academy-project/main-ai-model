"""조를 줄 목록으로 되돌린다. clean(글자 보존 계산)·chunk·validate가 같은 규칙을 쓴다."""

from __future__ import annotations

from .models import Article, Node

_INDENT = {"lead": "", "항": "", "호": "  ", "목": "    "}

_CIRCLED_START = 0x2460  # ①
_CIRCLED_20_END = 0x2473  # ⑳
_CIRCLED_21 = 0x3251  # ㉑
_CIRCLED_36 = 0x32B1  # ㊱


def circled_to_int(marker: str) -> int | None:
    if len(marker) != 1:
        return None
    code = ord(marker)
    if _CIRCLED_START <= code <= _CIRCLED_20_END:
        return code - _CIRCLED_START + 1
    if _CIRCLED_21 <= code <= 0x325F:
        return code - _CIRCLED_21 + 21
    if _CIRCLED_36 <= code <= 0x32BF:
        return code - _CIRCLED_36 + 36
    return None


def heading_line(article: Article) -> str:
    return article.label + (f"({article.title})" if article.title else "")


def node_first_line(node: Node) -> str:
    """노드 자기 줄(번호 + 글). 들여쓰기 없음."""
    return f"{node.marker} {node.text}".strip() if node.marker else node.text


def node_head_lines(node: Node) -> list[str]:
    """노드 자기 줄과 번호 없는 이어지는 줄(자식 제외)."""
    ind = _INDENT[node.kind]
    return [ind + node_first_line(node)] + [ind + ln for ln in node.lines]


def node_lines(node: Node) -> list[str]:
    """노드와 자손의 줄. 삭제된 노드는 건너뛴다."""
    if node.deleted:
        return []
    out = node_head_lines(node)
    for child in node.children:
        out.extend(node_lines(child))
    return out


def article_blocks(article: Article) -> list[Node]:
    """줄이 있는 최상위 노드(항 또는 본문)들."""
    return [n for n in article.nodes if node_lines(n)]


def article_lines(article: Article) -> list[str]:
    """조 전체 줄. 첫 노드가 원래 조 줄에 붙어 있었으면 조 제목 줄에 이어 붙인다."""
    head = heading_line(article)
    blocks = [node_lines(n) for n in article_blocks(article)]
    if not blocks:
        return [head]
    if article.inline_first:
        first = blocks[0]
        return [f"{head} {first[0].lstrip()}", *first[1:], *[ln for b in blocks[1:] for ln in b]]
    return [head, *[ln for b in blocks for ln in b]]
