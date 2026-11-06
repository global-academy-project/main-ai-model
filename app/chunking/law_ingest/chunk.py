"""S4 청킹: 정리된 조 → 조각(Piece).

규칙
  1. 조 1개 = 조각 1개. 짧은 조도 합치지 않는다(인용 단위가 조이다).
  2. 조 전체 글자 수가 max_chars를 넘을 때만 나눈다.
  3. 나눌 때는 항 경계를 먼저 쓴다. 항들을 상한까지 모아 한 조각으로 만든다.
  4. 한 항이 상한을 넘으면 호 경계에서 나눈다. 그 항의 도입 문장("… 다음 각 호의 …")을 뒤 조각 앞에 되풀이한다.
  5. 첫 조각 뒤의 모든 조각은 맨 앞에 조 제목 줄을 되풀이한다(어느 조의 일부인지 보이게).
  6. 호·목 중간에서는 자르지 않는다. 한 호가 상한을 넘으면 넘는 채로 두고 oversize로 표시한다.

되풀이한 줄 수는 `repeated_lead_lines`에 적는다. 그 줄을 빼고 이으면 원래 조와 같아야 한다(validate가 확인).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Article, Node, ParsedLaw, Piece
from .render import (
    article_blocks,
    article_lines,
    circled_to_int,
    heading_line,
    node_head_lines,
    node_lines,
)


@dataclass
class _Frag:
    """한 조각 안에 들어갈 수 있는 가장 작은 덩어리."""

    block: int  # 최상위 노드 번호
    lines: list[str]
    head: list[str]  # 이 덩어리가 항의 뒷부분일 때 되풀이할 도입부 줄(아니면 빈 목록)
    partial: bool  # 항의 일부만 담았는지


def _block_label(node: Node) -> str | None:
    if node.kind == "항":
        n = circled_to_int(node.marker)
        return f"제{n}항" if n else None
    if node.kind == "호":
        return f"제{node.marker.rstrip('.')}호"
    return None


def _text_len(lines: list[str]) -> int:
    return len("\n".join(lines))


def _split_block(node: Node, base_overhead: int, max_chars: int) -> list[_Frag] | None:
    """한 항이 너무 길 때 자식(호) 묶음 경계에서 나눈다. 더 못 나누면 None."""
    groups = [node_lines(c) for c in node.children if node_lines(c)]
    if len(groups) < 2:
        return None
    head = node_head_lines(node)
    frags: list[list[str]] = []
    cur: list[str] = []
    for g in groups:
        # 어느 조각이든 맨 앞에 [조 제목 줄 + 도입부]가 오는 경우를 기준으로 크기를 잰다.
        if cur and base_overhead + _text_len(head + cur + g) + 1 > max_chars:
            frags.append(cur)
            cur = []
        cur = cur + g
    if cur:
        frags.append(cur)
    out: list[_Frag] = []
    for i, body in enumerate(frags):
        if i == 0:
            out.append(_Frag(-1, head + body, [], partial=len(frags) > 1))
        else:
            out.append(_Frag(-1, body, head, partial=True))
    return out


def _make_frags(article: Article, max_chars: int) -> list[_Frag]:
    heading = heading_line(article)
    base = len(heading) + 1
    frags: list[_Frag] = []
    for bi, node in enumerate(article_blocks(article)):
        lines = node_lines(node)
        if base + _text_len(lines) <= max_chars:
            frags.append(_Frag(bi, lines, [], False))
            continue
        parts = _split_block(node, base, max_chars)
        if parts is None:
            frags.append(_Frag(bi, lines, [], False))
        else:
            for p in parts:
                p.block = bi
                frags.append(p)
    return frags


def _span(blocks: list[Node], frags: list[_Frag]) -> str | None:
    labels: list[str] = []
    partial = False
    for f in frags:
        lab = _block_label(blocks[f.block])
        if lab and lab not in labels:
            labels.append(lab)
        partial = partial or f.partial
    if not labels:
        return None
    text = labels[0] if len(labels) == 1 else f"{labels[0]}~{labels[-1]}"
    return text + (" 일부" if partial else "")


def split_article(article: Article, max_chars: int) -> list[Piece]:
    full = article_lines(article)
    if _text_len(full) <= max_chars:
        return [Piece(article.key, 0, 1, full, 0, None)]

    heading = heading_line(article)
    blocks = article_blocks(article)
    frags = _make_frags(article, max_chars)

    # 조각 묶기: 순서대로 담다가 상한을 넘으면 새 조각
    groups: list[list[_Frag]] = []
    cur: list[_Frag] = []

    def render(group: list[_Frag], first_piece: bool) -> tuple[list[str], int]:
        lines: list[str] = []
        repeated = 0
        if first_piece:
            body: list[str] = []
            for f in group:
                body.extend(f.lines)
            if article.inline_first and body:
                return [f"{heading} {body[0].lstrip()}", *body[1:]], 0
            return [heading, *body], 0
        lines.append(heading)
        repeated = 1
        if group[0].head:
            lines.extend(group[0].head)
            repeated += len(group[0].head)
        for f in group:
            lines.extend(f.lines)
        return lines, repeated

    for f in frags:
        trial = cur + [f]
        lines, _ = render(trial, first_piece=not groups)
        if cur and _text_len(lines) > max_chars:
            groups.append(cur)
            cur = [f]
        else:
            cur = trial
    if cur:
        groups.append(cur)

    pieces: list[Piece] = []
    for i, g in enumerate(groups):
        lines, repeated = render(g, first_piece=(i == 0))
        pieces.append(
            Piece(
                article_key=article.key,
                part=i,
                parts_total=len(groups),
                lines=lines,
                repeated_lead_lines=repeated,
                span=_span(blocks, g),
                oversize=_text_len(lines) > max_chars,
            )
        )
    return pieces


def make_pieces(law: ParsedLaw, max_chars: int) -> list[Piece]:
    pieces: list[Piece] = []
    for a in law.articles:
        if a.status != "ACTIVE":
            continue
        pieces.extend(split_article(a, max_chars))
    return pieces
