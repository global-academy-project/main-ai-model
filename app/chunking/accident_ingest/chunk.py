"""S4 청킹: 사례 1개 = 청크 1개(D8). 넘칠 때만 섹션 경계에서 나눈다.

합본 모양:
    [재해개요]
    {본문}

    [재해유발요인]
    {본문}

    [위험성 감소대책(예시)]
    {본문}
라벨은 우리가 붙이는 구조 표시이고 원문 글자는 바꾸지 않는다. 비어 있는 섹션은 라벨째 만들지 않는다.

넘칠 때(합본 > max_chars):
  섹션마다 따로 조각을 만든다(슬롯 0/1/2). 한 섹션도 넘치면 줄(▶ 항목) → 문장 끝 순으로 더 나눈다.
  두 번째 조각부터 라벨 줄을 되풀이하고 repeated_lead_lines 에 적는다(되풀이한 줄을 빼고 이으면 원문).
  더 쪼갤 수 없는 한 줄이 상한을 넘으면 자르지 않고 oversize 로 표시한다.
"""

from __future__ import annotations

import re
from typing import Any

from . import config
from .config import BuildSettings

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def label_line(label: str) -> str:
    return f"[{label}]"


def sections_of(clean: dict[str, str]) -> list[tuple[str, str, str]]:
    """비어 있지 않은 섹션: [(슬롯 이름, 라벨, 본문)]"""
    return [(name, label, clean[field]) for name, field, label in config.SECTIONS if clean.get(field)]


def case_text(sections: list[tuple[str, str, str]]) -> str:
    return "\n\n".join(f"{label_line(label)}\n{body}" for _, label, body in sections)


def _units(body: str, budget: int) -> list[tuple[str, bool]]:
    """본문을 나눌 수 있는 단위(줄, 줄이 길면 문장)로 쪼갠다. (텍스트, 상한 초과 여부)"""
    out: list[tuple[str, bool]] = []
    for ln in body.split("\n"):
        if not ln.strip():
            continue
        if len(ln) <= budget:
            out.append((ln, False))
            continue
        for sent in _SENT_RE.split(ln):
            if sent:
                out.append((sent, len(sent) > budget))
    return out


def _section_pieces(name: str, label: str, body: str, max_chars: int) -> list[dict[str, Any]]:
    lead = label_line(label)
    slot = config.SLOT[name]
    full = f"{lead}\n{body}"
    if len(full) <= max_chars:
        return [{"section": name, "slot": slot, "lines": full.split("\n"), "repeated_lead_lines": 0, "oversize": False}]
    budget = max_chars - len(lead) - 1
    groups: list[list[tuple[str, bool]]] = []
    cur: list[tuple[str, bool]] = []
    size = 0
    for unit in _units(body, budget):
        add = len(unit[0]) + (1 if cur else 0)
        if cur and size + add > budget:
            groups.append(cur)
            cur, size = [], 0
            add = len(unit[0])
        cur.append(unit)
        size += add
    if cur:
        groups.append(cur)
    pieces = []
    for i, g in enumerate(groups):
        pieces.append(
            {
                "section": name,
                "slot": slot,
                "lines": [lead, *[u[0] for u in g]],
                "repeated_lead_lines": 0 if i == 0 else 1,
                "oversize": any(u[1] for u in g),
            }
        )
    return pieces


def make_pieces(doc: dict[str, Any], settings: BuildSettings) -> list[dict[str, Any]]:
    """정리된 문서 하나 → 조각 목록. 서술이 하나도 없으면 빈 목록."""
    sections = sections_of(doc["clean"])
    if not sections:
        return []
    combined = case_text(sections)
    if len(combined) <= settings.max_chars:
        raw_pieces = [{"section": "case", "slot": config.SLOT["case"], "lines": combined.split("\n"), "repeated_lead_lines": 0, "oversize": False}]
    else:
        raw_pieces = [p for name, label, body in sections for p in _section_pieces(name, label, body, settings.max_chars)]
    # part 번호는 섹션(슬롯) 안에서 0부터. parts_total 은 같은 섹션의 조각 수.
    totals: dict[str, int] = {}
    for p in raw_pieces:
        totals[p["section"]] = totals.get(p["section"], 0) + 1
    counters: dict[str, int] = {}
    out = []
    for p in raw_pieces:
        part = counters.get(p["section"], 0)
        counters[p["section"]] = part + 1
        out.append({"external_id": doc["external_id"], **p, "part": part, "parts_total": totals[p["section"]]})
    return out
