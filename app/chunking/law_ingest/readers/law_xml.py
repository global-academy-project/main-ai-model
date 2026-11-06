"""S1 읽기: 국가법령정보센터 Open API 본문 XML(lawService.do ... type=XML) → SourceDoc.

태그 계층(법제처 공동활용 법령 본문 응답):
  <법령>
    <기본정보> 법령명_한글, 법령명약칭, 시행일자, 공포일자, 공포번호, 법종구분, 소관부처, 제개정구분 …
    <조문>
      <조문단위>  조문번호, 조문가지번호, 조문여부(조문|전문), 조문제목, 조문내용, 조문이동이전/이후 …
        <항>  항번호, 항내용
          <호>  호번호, 호내용
            <목>  목번호, 목내용
    <부칙> <부칙단위> 부칙내용
    <별표> …            (읽지 않는다: 별표는 청크 대상이 아니다)

hwpx reader와 같은 모양의 문단(한 줄 = 한 문단)으로 내보내므로 parse 이후 단계는 고치지 않는다.
  편·장·절·관·속 제목  ← 조문여부가 '전문'인 조문단위(또는 내용이 "제N장 …" 꼴인 것)
  조                  ← 조문내용. 제목만 있으면 첫 항을 같은 줄에 붙인다(hwpx 본문과 같은 모양)
  항·호·목            ← 항내용·호내용·목내용. 번호가 내용에 없으면 번호 태그를 앞에 붙인다
  부칙                ← "부칙 <…>"으로 시작하는 줄

확인한 것: 위 태그 이름·계층(법제처 응답 구조 공개 자료). 실제 응답 파일로 확인 못 한 것: 조문내용에 번호가 들어 있는지,
'이동' 조의 표기 위치, 법종구분 값의 모양. 코드는 번호가 있든 없든, 내용이 겹치든 말든 견디게 썼다(겹치면 한 번만).
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..models import RawParagraph, SourceDoc, SourceMeta
from .hwpx import ReaderError

_MAX_BYTES = 64 * 1024 * 1024
_HEADING_TEXT_RE = re.compile(r"^\s*제\d+(?:의\d+)?(?:편|장|절|관|속)(?:\s|$)")
_ARTICLE_HEAD_RE = re.compile(r"^\s*제\d+조(?:의\d+)?(?=[\s(<]|$)")
_TITLE_ONLY_RE = re.compile(r"^제\d+조(?:의\d+)?\s*\([^()]*\)\s*$")
_NOTE_LINE_RE = re.compile(r"^\[[^\[\]]{1,200}\]$")  # [본조신설 …], [제N조에서 이동 …]
_INDENT = {"항": "  ", "호": "  ", "목": "    "}


def _t(el: ET.Element | None, name: str) -> str:
    if el is None:
        return ""
    child = el.find(name)
    return "".join(child.itertext()).strip() if child is not None else ""


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.replace("\r", "").replace(" ", " ").split("\n") if ln.strip()]


def _squash(text: str) -> str:
    return "".join(text.split())


def _date(value: str) -> str | None:
    digits = re.sub(r"\D", "", value or "")
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) >= 8 else None


def _meta(root: ET.Element, path: Path, data: bytes) -> SourceMeta:
    info = root.find("기본정보")
    name = _t(info, "법령명_한글") or path.stem
    kind, dept, no = _t(info, "법종구분"), _t(info, "소관부처"), _t(info, "공포번호")
    if kind == "부령" and dept:
        kind = f"{dept}령"
    promul_no = f"{kind} 제{no}호".strip() if no else None
    return SourceMeta(
        law_name=name,
        law_alias=_t(info, "법령명약칭") or None,
        effective_date=_date(_t(info, "시행일자")),
        promulgation_no=promul_no,
        promulgation_date=_date(_t(info, "공포일자")),
        amendment_type=_t(info, "제개정구분") or None,
        source_file=path.name,
        source_format="xml",
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


def _with_marker(marker: str, content: str) -> str:
    """내용이 번호로 시작하지 않으면 번호를 앞에 붙인다."""
    content = content.strip()
    marker = marker.strip()
    if not marker or content.startswith(marker):
        return content
    return f"{marker} {content}".strip()


def _child_lines(parent: ET.Element, out: list[str]) -> None:
    """항·호·목을 문서 순서대로 한 줄씩 낸다. 항 아래 호, 호 아래 목. 조 바로 아래 호도 받는다."""
    for node in parent:
        if node.tag not in {"항", "호", "목"}:
            continue
        content = _t(node, f"{node.tag}내용")
        marker = _t(node, f"{node.tag}번호")
        parts = _lines(content)
        if parts:
            parts[0] = _with_marker(marker, parts[0])
            out.append(_INDENT[node.tag] + parts[0])
            out.extend(_INDENT[node.tag] + p for p in parts[1:])
        elif marker:
            out.append(_INDENT[node.tag] + marker)
        _child_lines(node, out)


def _article_lines(unit: ET.Element) -> list[str]:
    number = _t(unit, "조문번호")
    branch = _t(unit, "조문가지번호")
    title = _t(unit, "조문제목")
    label = f"제{number}조" + (f"의{branch}" if branch else "")
    body = _lines(_t(unit, "조문내용"))

    if not body:
        head = f"{label}({title})" if title else label
    elif _ARTICLE_HEAD_RE.match(body[0]):
        head = body[0]
    else:  # 번호·제목이 내용에 없는 응답
        head = f"{label}({title}) {body[0]}".strip() if title else f"{label} {body[0]}"
    # 별행 주기([본조신설 …] 등)는 hwpx 본문처럼 조의 맨 끝으로 보낸다(위치와 상관없이 같은 조 안에서 읽힌다).
    notes = [ln for ln in body[1:] if _NOTE_LINE_RE.match(ln)]
    rest = [ln for ln in body[1:] if not _NOTE_LINE_RE.match(ln)]

    children: list[str] = []
    _child_lines(unit, children)

    # 조문내용이 항 내용을 이미 품고 있으면(겹침) 한 번만 쓴다.
    have = _squash(head + "".join(rest))
    kept = [ln for ln in children if _squash(ln) not in have or len(_squash(ln)) < 6]

    # hwpx 본문은 제목 줄 뒤에 첫 항이 같은 줄로 이어진다. 같은 모양으로 맞춘다.
    if not rest and kept and _TITLE_ONLY_RE.match(head.strip()) and kept[0].lstrip()[:1] in "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳":
        head = f"{head.strip()} {kept[0].strip()}"
        kept = kept[1:]
    return [head, *rest, *kept, *notes]


def read_law_xml(path: str | Path) -> SourceDoc:
    path = Path(path)
    data = path.read_bytes()
    if len(data) > _MAX_BYTES:
        raise ReaderError("XML 파일이 너무 큽니다.")
    head = data[:2048].lstrip(b"\xef\xbb\xbf").lstrip()
    if not head.startswith(b"<"):
        raise ReaderError(f"XML이 아닙니다(앞부분: {head[:60].decode('utf-8', 'replace')!r}). 인증 실패·오류 응답이 저장된 것은 아닌지 확인하세요.")
    if b"<!DOCTYPE" in data[:4096].upper() or b"<!ENTITY" in data.upper():
        raise ReaderError("DOCTYPE/ENTITY가 있는 XML은 읽지 않습니다.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ReaderError(f"XML을 해석하지 못했습니다: {exc}") from exc

    if root.tag != "법령":
        found = root.find(".//법령")
        if found is None:
            text = "".join(root.itertext()).strip()[:80]
            raise ReaderError(f"법령 본문 XML이 아닙니다(루트 <{root.tag}>: {text!r}). lawService.do ... type=XML 응답이어야 합니다.")
        root = found
    units = root.findall("./조문/조문단위")
    if not units:
        raise ReaderError("<조문단위>가 없습니다. 법령 본문 조회(lawService.do) 응답이 맞는지 확인하세요.")

    texts: list[str] = []
    for unit in units:
        content = _t(unit, "조문내용")
        is_heading = _t(unit, "조문여부") == "전문" or bool(_HEADING_TEXT_RE.match(content) and not _ARTICLE_HEAD_RE.match(content))
        if is_heading:
            texts.extend(_lines(content))
        else:
            texts.extend(_article_lines(unit))

    for add in root.findall("./부칙/부칙단위"):
        body = _lines(_t(add, "부칙내용"))
        if not body:
            continue
        if not body[0].lstrip().startswith("부칙"):
            no, day = _t(add, "부칙공포번호"), _date(_t(add, "부칙공포일자")) or ""
            body.insert(0, f"부칙 <제{no}호, {day}>" if no else "부칙")
        texts.extend(body)

    paragraphs = [RawParagraph(i, t) for i, t in enumerate(texts)]
    return SourceDoc(meta=_meta(root, path, data), paragraphs=paragraphs)
