"""S1 읽기: .hwpx → SourceDoc.

.hwpx는 zip 안의 XML이다. 본문은 Contents/section*.xml의 최상위 문단(<hp:p>)이고,
글자는 <hp:t>에 들어 있다. 외부 라이브러리 없이 표준 라이브러리만 쓴다.

여기서는 글자를 바꾸지 않는다(앞 공백도 그대로). 줄바꿈 요소만 '\\n'으로 옮긴다.
조·항·호 구분은 다음 단계(parse)가 글자 모양으로 한다.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ..models import RawParagraph, SourceDoc, SourceMeta

_SECTION_RE = re.compile(r"Contents/section(\d+)\.xml")
_MAX_ENTRY_BYTES = 64 * 1024 * 1024  # 압축 폭탄 방어용 상한


class ReaderError(ValueError):
    """읽을 수 없는 파일."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _t_text(t: ET.Element) -> str:
    parts = [t.text or ""]
    for child in t:
        name = _local(child.tag)
        if name == "lineBreak":
            parts.append("\n")
        elif name == "tab":
            parts.append("\t")
        elif name in {"fwSpace", "nbSpace"}:
            parts.append(" ")
        parts.append(child.tail or "")
    return "".join(parts)


def paragraph_text(p: ET.Element) -> str:
    """문단 안 모든 <t>를 문서 순서대로 이어 붙인다(표·머리말 안 글자도 포함)."""
    return "".join(_t_text(el) for el in p.iter() if _local(el.tag) == "t")


_EFFECTIVE_RE = re.compile(r"\[\s*시행\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*\]")
_PROMUL_RE = re.compile(
    r"\[\s*([^\[\]]*?)\s*제\s*(\d+)\s*호\s*,\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\s*,\s*([^\[\]]+?)\s*\]"
)
_ALIAS_RE = re.compile(r"^(.*?)\s*\(\s*약칭\s*:?\s*(.*?)\s*\)\s*$")


def _date(y: str, m: str, d: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def extract_meta(texts: list[str], source_file: str) -> SourceMeta:
    """문서 첫머리(처음 30개 문단)에서 법령명·시행일·공포번호를 읽는다. 못 읽으면 None."""
    head = [t for t in texts[:30]]
    joined = "\n".join(head)

    name = ""
    alias = None
    for t in head:
        s = t.strip()
        if not s or "[시행" in s:
            continue
        m = _ALIAS_RE.match(s)
        name, alias = (m.group(1), m.group(2)) if m else (s, None)
        break
    if not name:
        name = Path(source_file).stem

    effective = promul_no = promul_date = amend = None
    m = _EFFECTIVE_RE.search(joined)
    if m:
        effective = _date(*m.groups())
    m = _PROMUL_RE.search(joined)
    if m:
        kind, no, y, mo, d, amend = m.groups()
        promul_no = f"{kind} 제{no}호".strip()
        promul_date = _date(y, mo, d)

    return SourceMeta(
        law_name=name,
        law_alias=alias,
        effective_date=effective,
        promulgation_no=promul_no,
        promulgation_date=promul_date,
        amendment_type=amend,
    )


def read_hwpx(path: str | Path) -> SourceDoc:
    path = Path(path)
    data = path.read_bytes()
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ReaderError(f"hwpx(zip) 파일이 아닙니다: {path.name}") from exc

    with zf:
        sections = sorted(
            (n for n in zf.namelist() if _SECTION_RE.fullmatch(n)),
            key=lambda n: int(_SECTION_RE.fullmatch(n).group(1)),  # type: ignore[union-attr]
        )
        if not sections:
            raise ReaderError("Contents/section*.xml이 없습니다. hwpx가 맞는지 확인하세요.")
        paragraphs: list[RawParagraph] = []
        for name in sections:
            if zf.getinfo(name).file_size > _MAX_ENTRY_BYTES:
                raise ReaderError(f"{name} 크기가 너무 큽니다.")
            root = ET.fromstring(zf.read(name))
            for p in root:
                if _local(p.tag) == "p":
                    paragraphs.append(RawParagraph(len(paragraphs), paragraph_text(p)))

    meta = extract_meta([p.text for p in paragraphs], path.name)
    meta.source_file = path.name
    meta.source_format = "hwpx"
    meta.source_sha256 = hashlib.sha256(data).hexdigest()
    return SourceDoc(meta=meta, paragraphs=paragraphs)
