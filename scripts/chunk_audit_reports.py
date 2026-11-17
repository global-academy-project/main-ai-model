#!/usr/bin/env python3
"""감사 보고서(PDF)를 청크 JSONL 로 만든다. 산출물은 scripts/load_chunks.py 의 입력이다.

파이프라인: 추출(pdfplumber) -> 정리 -> 지적사항 분할 -> 크기 제한 청킹 -> 필드 채우기 -> 검증 -> 저장.
지적사항 하나 = 원본(source) 하나이므로, 같은 지적사항의 청크는 external_id 가 같다.

지적사항 시작 표지는 문서마다 다르다. 문서별로 아래 둘 중 더 많이 잡히는 쪽을 자동으로 고른다.
  title     '제 목 ...', '【제 목】 ...' 줄 (일련번호·처분 종류 줄이 바로 앞에 붙은 것까지 포함)
  circled   '① 제목' 줄 바로 뒤(3줄 안)에 '【 문제점 】' 이 오는 형식
어느 쪽도 2건 이상 잡히지 않으면 지적사항 분할을 하지 않고, 문서 전체를 장·절 순서의 청크로 만든다.
지적사항이 잡힌 문서에서 첫 지적사항 앞 부분(감사 개요 등)은 기본으로 넣지 않는다(--overview include 로 넣는다).

분류 태그(accident_type, hazard, work_type, cas, ghs)는 자동으로 추측하지 않고 빈 배열로 둔다.
산출물은 사람이 눈으로 확인한 뒤 load_chunks.py 로 검증·적재한다.

종료 코드: 0 성공, 2 입력·설정 오류.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_MAX_CHARS = 1500
DEFAULT_MIN_CHARS = 250
DEFAULT_MAX_CHUNKS = 8  # 임베딩 서버가 한 요청에서 받는 청크 수(EMBEDDING_MAX_BATCH_SIZE 기본값)
SOURCE_TYPE = "ACCIDENT_REPORT"
EMPTY_TAGS = ("accident_type", "hazard", "work_type", "cas", "ghs")
DISPOSITIONS = ("통보", "주의", "시정", "개선", "모범", "징계", "경고", "권고", "회수", "감액", "변상")

# ---------------------------------------------------------------- 정규식
FOOTER_RE = re.compile(r"^-\s*\d+\s*-$")
TOC_RE = re.compile(r"(·{3,}|\.{5,}|…{2,})")
WATERMARK_RE = re.compile(r"(INSIDabcdef|fedcbaDISNI|MS_\d+MS_\d+|SM\d+_SM)")
TITLE_RE = re.compile(r"^(?:【\s*제\s*목\s*】|제\s?목(?!\s*순번))\s*:?\s*(.*)$")
CIRCLED_RE = re.compile(r"^([①-⑳])\s+(\S.*)$")
PROBLEM_RE = re.compile(r"^【\s*문제점\s*】")
ROMAN_RE = re.compile(r"^([Ⅰ-Ⅻ])\.?\s*(\S.*)?$")
BOX_RE = re.compile(r"^[□■▣◆◇]\s*(\S.*)$")
MARKER_RE = re.compile(r"^([○❍●ㅇ▶▷◈◦※\-\*·•□■【\[]|[①-⑳]|[가-하]\.|\d+[\.\)]\s|\(\d+\)|\([가-하]\))")
SENT_END_RE = re.compile(r"(?:[.!?。다요음함]\s?[.)」』”’]*|[.)])$")
HEADER_LOOKBACK = 8


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- 1. 추출
class InputError(Exception):
    pass


def extract_pages(path: Path) -> list[str]:
    """PDF -> 쪽별 텍스트. .txt 는 이미 추출된 텍스트로 보고 '\\n\\f\\n' 로 쪽을 나눈다."""
    if path.suffix.lower() == ".txt":
        try:
            return path.read_text(encoding="utf-8-sig").split("\n\f\n")
        except (OSError, UnicodeDecodeError) as e:
            raise InputError(f"{path}: 읽을 수 없다 ({e})")
    try:
        import pdfplumber  # 지연 임포트: txt 입력·시험에서는 필요 없다
    except ImportError:
        raise InputError("pdfplumber 가 없다. 'pip install pdfplumber' 로 설치한다")
    try:
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for pg in pdf.pages:
                # 겹쳐 찍힌 글자(제목 장식 등)를 하나로 줄인다
                pages.append(pg.dedupe_chars(tolerance=1).extract_text() or "")
        return pages
    except FileNotFoundError:
        raise InputError(f"{path}: 파일이 없다")
    except Exception as e:  # 손상·암호화 PDF
        raise InputError(f"{path}: PDF 를 읽지 못했다 ({type(e).__name__}: {e})")


# ---------------------------------------------------------------- 2. 정리
def undouble(line: str) -> str:
    """'현현장장 안안전전' 처럼 글자마다 겹쳐 찍힌 줄을 되돌린다(어절 대부분이 그럴 때만)."""
    toks = line.split(" ")

    def is_dbl(t: str) -> bool:
        return len(t) >= 2 and len(t) % 2 == 0 and t[0::2] == t[1::2]

    cand = [t for t in toks if len(t) >= 2]
    if len(cand) < 2 or sum(is_dbl(t) for t in cand) / len(cand) < 0.8:
        return line
    return " ".join(t[0::2] if is_dbl(t) else t for t in toks)


def clean_lines(pages: list[str]):
    """쪽 텍스트 -> [(쪽 번호(1부터), 줄)]. 쪽 번호표·목차 점선·워터마크 줄은 버린다."""
    out = []
    for pno, text in enumerate(pages, 1):
        for raw in text.replace("\r", "").split("\n"):
            line = undouble(re.sub(r"[ \t\u00a0\u3000]+", " ", raw).strip())
            if line and (FOOTER_RE.match(line) or TOC_RE.search(line) or WATERMARK_RE.search(line)):
                continue
            out.append((pno, line))
    return out


# ---------------------------------------------------------------- 3. 분할
def _is_header_line(line: str) -> bool:
    """제목 앞에 붙는 머리 줄인가: 짧고, 본문 글머리표로 시작하지 않는다."""
    if not line or len(line) > 40 or heading_of(line):
        return False
    return not re.match(r"^([○❍●▶▷◈◦※\-\*·•]|\d+[\.\)]\s)", line)


def _find_title_blocks(lines):
    """title 방식 시작 위치 [(시작 줄, 제목 줄, 제목)]."""
    found = []
    for i, (_, line) in enumerate(lines):
        m = TITLE_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip()
        if not title:  # 제목이 다음 줄로 넘어간 경우
            j = i + 1
            while j < len(lines) and not lines[j][1]:
                j += 1
            title = lines[j][1] if j < len(lines) else ""
        start = i
        j = i - 1
        # 제목 바로 앞의 머리 줄(일련번호·처분 종류·기관명)만 같은 쪽 안에서 함께 묶는다
        while j >= 0 and i - j <= HEADER_LOOKBACK and lines[j][0] == lines[i][0] and _is_header_line(lines[j][1]):
            start = j
            j -= 1
        found.append((start, i, title))
    return found


def _find_circled_blocks(lines):
    found = []
    for i, (_, line) in enumerate(lines):
        m = CIRCLED_RE.match(line)
        if not m:
            continue
        window = [l for _, l in lines[i + 1:i + 5]]
        if any(PROBLEM_RE.match(w) for w in window):
            found.append((i, i, m.group(2).strip()))
    return found


STRATEGIES = (("title", _find_title_blocks), ("circled", _find_circled_blocks))


def detect(lines):
    """가장 많이 잡히는 방식. 2건 미만이면 (None, [])."""
    best_name, best = None, []
    for name, fn in STRATEGIES:
        hits = fn(lines)
        if len(hits) > len(best):
            best_name, best = name, hits
    if len(best) < 2:
        return None, []
    # 겹치는 시작 위치(앞 머리줄 탐색이 이전 제목 줄까지 닿은 경우)를 바로잡는다
    fixed, prev_title_line = [], -1
    for start, tline, title in best:
        fixed.append((max(start, prev_title_line + 1), tline, title))
        prev_title_line = tline
    return best_name, fixed


def heading_of(line: str):
    m = ROMAN_RE.match(line)
    if m and m.group(2):
        return f"{m.group(1)} {m.group(2).strip()}"
    return None


def split_findings(lines, hits):
    """지적사항 블록 목록과 앞부분(overview) 줄.

    장 제목(Ⅰ. 설계단계 등) 줄은 본문에 넣지 않고, 그 뒤 블록들의 section 으로만 쓴다.
    """
    starts = [h[0] for h in hits]
    overview = lines[:starts[0]]
    section = ""
    for _, line in overview:
        section = heading_of(line) or section
    blocks = []
    for k, (start, _tline, title) in enumerate(hits):
        end = starts[k + 1] if k + 1 < len(hits) else len(lines)
        body, block_section = [], section
        for pno, line in lines[start:end]:
            h = heading_of(line)
            if h:
                section = h
                continue
            body.append((pno, line))
        blocks.append({"title": title, "lines": body, "section": block_section})
    return blocks, overview


def split_sections(lines):
    """지적사항 방식이 없을 때: □·Ⅰ 제목 단위로 나눈다. [(제목, 줄들)]"""
    sections, cur_title, cur = [], "", []
    for pno, line in lines:
        h = heading_of(line)
        b = BOX_RE.match(line)
        if h or b:
            if any(l for _, l in cur):
                sections.append((cur_title, cur))
            cur_title = h or b.group(1).strip()
            cur = [(pno, line)]
        else:
            cur.append((pno, line))
    if any(l for _, l in cur):
        sections.append((cur_title, cur))
    return sections


# ---------------------------------------------------------------- 4. 문단·청킹
def _digit_heavy(line: str) -> bool:
    chars = [c for c in line if not c.isspace()]
    return len(chars) >= 4 and sum(c.isdigit() or c in ",.-%()" for c in chars) / len(chars) > 0.5


def to_paragraphs(lines):
    """[(쪽, 줄)] -> [(쪽, 문단)]. 문단 안에서 줄바꿈은 공백으로 잇는다."""
    paras, cur, cur_page, prev = [], [], None, ""
    for pno, line in lines:
        if not line:
            if cur:
                paras.append((cur_page, " ".join(cur)))
                cur, cur_page = [], None
            prev = ""
            continue
        brk = (not cur) or bool(MARKER_RE.match(line)) or _digit_heavy(line) or _digit_heavy(prev) \
            or bool(SENT_END_RE.search(prev))
        if brk and cur:
            paras.append((cur_page, " ".join(cur)))
            cur, cur_page = [], None
        if not cur:
            cur_page = pno
        cur.append(line)
        prev = line
    if cur:
        paras.append((cur_page, " ".join(cur)))
    return paras


def _split_long(text: str, max_chars: int):
    """한 문단이 max_chars 보다 길면 문장 끝에서, 그래도 길면 글자 수로 자른다."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.다])\s+", text):
        while len(sent) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_paragraphs(paras, max_chars: int, min_chars: int):
    """[(쪽, 문단)] -> [{'text', 'page_start', 'page_end'}]. 문단 경계를 지키며 max_chars 이하로 묶는다."""
    pieces = []
    for page, text in paras:
        for part in _split_long(text, max_chars):
            pieces.append((page, part))
    chunks, cur, p0, p1 = [], "", None, None
    for page, part in pieces:
        if cur and len(cur) + 1 + len(part) > max_chars:
            chunks.append({"text": cur, "page_start": p0, "page_end": p1})
            cur, p0 = "", None
        cur = f"{cur}\n{part}" if cur else part
        p0 = page if p0 is None else p0
        p1 = page
    if cur:
        chunks.append({"text": cur, "page_start": p0, "page_end": p1})
    # 끝의 너무 짧은 조각은 앞 청크에 붙인다(붙여도 상한의 125% 이내일 때만)
    if len(chunks) >= 2 and len(chunks[-1]["text"]) < min_chars \
            and len(chunks[-2]["text"]) + 1 + len(chunks[-1]["text"]) <= int(max_chars * 1.25):
        last = chunks.pop()
        chunks[-1]["text"] += "\n" + last["text"]
        chunks[-1]["page_end"] = last["page_end"]
    return chunks


def partition(items, max_n: int):
    """items 를 max_n 개 이하의 묶음으로 균등하게 나눈다(순서 유지, 앞 묶음이 하나 더 많을 수 있다)."""
    n = len(items)
    if n <= max_n:
        return [list(items)]
    parts = -(-n // max_n)
    base, extra = divmod(n, parts)
    out, i = [], 0
    for k in range(parts):
        size = base + (1 if k < extra else 0)
        out.append(list(items[i:i + size]))
        i += size
    return out


def part_ident(ext_id: str, title: str, k: int, total: int):
    """나뉜 묶음의 (external_id, title). 나뉘지 않으면 그대로."""
    if total == 1:
        return ext_id, clip(title, 300)
    suffix = f" ({k}/{total})"
    return f"{ext_id}-P{k}", clip(title, 300 - len(suffix)) + suffix


def merge_small(chunks, max_chars: int, min_chars: int):
    """너무 짧은 청크(표지·제목만 있는 절)를 다음 청크에 붙인다(합쳐도 상한 이하일 때만)."""
    out, i = [], 0
    while i < len(chunks):
        cur = dict(chunks[i])
        while len(cur["text"]) < min_chars and i + 1 < len(chunks) \
                and len(cur["text"]) + 1 + len(chunks[i + 1]["text"]) <= max_chars:
            nxt = chunks[i + 1]
            cur["text"] += "\n" + nxt["text"]
            cur["page_end"] = nxt["page_end"]
            i += 1
        out.append(cur)
        i += 1
    return out


# ---------------------------------------------------------------- 5. 필드
def _ordered_tokens(text: str) -> list[str]:
    """text 안의 처분 종류를 나타나는 순서대로(중복 없이)."""
    hits = sorted((text.find(d), d) for d in DISPOSITIONS if d in text)
    return [d for _, d in hits]


def disposition_of(block_lines) -> list[str]:
    """처분 종류. 머리 줄(제목 앞뒤)에서 먼저 찾고, 없으면 본문의 '(통보)' 꼴을 쓴다."""
    head = " ".join(compact(l) for _, l in block_lines[:14] if len(compact(l)) <= 60)
    found = _ordered_tokens(head)
    if found:
        return found
    body = " ".join(l for _, l in block_lines)
    inner = " ".join(compact(m.group(1)) for m in re.finditer(r"\(([^()]{1,12})\)", body))
    return [d for d in _ordered_tokens(inner) if re.search(rf"(^|\s){d}", inner)]


def doc_defaults_from_name(path: Path):
    stem = path.stem
    m = re.match(r"^(\d+)[_\-\s\.\+]*(.*)$", stem)
    doc_id = m.group(1) if m else sha256_hex(stem)[:6]
    rest = m.group(2) if m else stem
    title = re.sub(r"[_\+]+", " ", rest)
    title = re.sub(r"\((?:공개|공개용|마스킹|홈페이지)[^)]*\)|\(\d+\)", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .-_") or stem
    ym = re.search(r"(20\d{2})", stem)
    return {"doc_id": doc_id, "doc_title": title, "year": int(ym.group(1)) if ym else None}


def load_manifest(path: Path | None):
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InputError(f"{path}: manifest 를 읽을 수 없다 ({e})")
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise InputError(f"{path}: manifest 는 {{파일명 또는 번호: {{doc_id, title, publisher, year, url, license}}}} 객체여야 한다")
    return data


def manifest_entry(manifest: dict, path: Path, doc_id: str):
    for key in (path.name, path.stem, doc_id):
        if key in manifest:
            return manifest[key]
    return {}


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def make_locator(p0, p1, label=""):
    pages = f"p{p0}" if p0 == p1 else f"p{p0}-{p1}"
    return clip(f"{pages} {label}".strip(), 100)


def build_source(ext_id, title, path, meta, common, text_for_hash):
    src = {"source_type": SOURCE_TYPE, "external_id": ext_id, "title": clip(title, 300),
           "file_path": clip(path.name, 500), "content_sha256": sha256_hex(text_for_hash), "flagged": False}
    for key in ("publisher", "license"):
        val = meta.get(key) or common.get(key)
        if val:
            src[key] = val
    if meta.get("url"):
        src["url"] = meta["url"]
    return src


def emit_source(src, doc_title, section, chunks, payload_base, minimal, prefix_parts):
    """원본 하나 -> JSONL 행 목록."""
    rows = []
    for idx, ch in enumerate(chunks):
        header = " > ".join(p for p in [doc_title, section, *prefix_parts] if p)
        payload = {k: [] for k in EMPTY_TAGS}
        if not minimal:
            payload.update(payload_base)
            payload["chunk_pages"] = [ch["page_start"], ch["page_end"]]
        row = dict(src)
        row.update({
            "chunk_index": idx,
            "text": ch["text"],
            "embedding_text": clip(f"{header}\n{ch['text']}" if header else ch["text"], 32768),
            "locator": make_locator(ch["page_start"], ch["page_end"], clip(section, 40) if section else ""),
            "payload": payload,
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 6. 문서 하나 처리
def process_document(path: Path, manifest: dict, common: dict, opts) -> dict:
    pages = extract_pages(path)
    lines = clean_lines(pages)
    if not any(l for _, l in lines):
        raise InputError(f"{path.name}: 추출된 텍스트가 없다(스캔본이면 OCR 이 필요하다)")
    d = doc_defaults_from_name(path)
    meta = manifest_entry(manifest, path, d["doc_id"])
    doc_id = str(meta.get("doc_id") or d["doc_id"])
    doc_title = str(meta.get("title") or d["doc_title"])
    year = meta.get("year", d["year"])
    if year is not None and not isinstance(year, int):
        raise InputError(f"{path.name}: manifest 의 year 는 정수여야 한다")
    strategy, hits = detect(lines)

    rows, stats = [], {"file": path.name, "doc_id": doc_id, "pages": len(pages), "strategy": strategy or "sections",
                       "findings": 0, "chunks": 0, "overview_chunks": 0, "sources": 0, "split_sources": 0}
    base = {"doc_id": doc_id, "doc_title": doc_title, "source_file": path.name}
    if year is not None:
        base["year"] = year

    def add(src_id, title, section, chunks, kind, extra=None, prefix=()):
        # 청크가 max_chunks_per_source 를 넘으면 원본을 여러 개(-P1, -P2 …)로 나눈다
        parts = partition(chunks, opts.max_chunks_per_source)
        for k, part in enumerate(parts, 1):
            pid, ptitle = part_ident(src_id, title, k, len(parts))
            src = build_source(pid, ptitle, path, meta, common, "\n".join(c["text"] for c in part))
            payload = dict(base, kind=kind, **(extra or {}))
            pre = [ptitle if x == title else x for x in prefix]
            rows.extend(emit_source(src, doc_title, section, part, payload, opts.minimal_payload, pre))
        stats["sources"] += len(parts)
        stats["split_sources"] += len(parts) > 1

    if strategy:
        blocks, overview = split_findings(lines, hits)
        for n, b in enumerate(blocks, 1):
            paras = to_paragraphs(b["lines"])
            chunks = chunk_paragraphs(paras, opts.max_chars, opts.min_chars)
            if not chunks:
                continue
            extra = {"finding_no": n}
            disp = disposition_of(b["lines"])
            if disp:
                extra["disposition"] = disp
            # 앞선 장 제목은 문서에 따라 지적사항과 무관할 수 있어 --use-headings 일 때만 쓴다
            section = b["section"] if opts.use_headings else ""
            if section:
                extra["section"] = section
            add(f"AUD-{doc_id}-{n:03d}", b["title"] or f"{doc_title} 지적사항 {n}", section, chunks,
                "finding", extra, prefix=[b["title"]])
            stats["findings"] += 1
        if opts.overview == "include":
            chunks = chunk_paragraphs(to_paragraphs(overview), opts.max_chars, opts.min_chars)
            if chunks:
                add(f"AUD-{doc_id}-OVW", f"{doc_title} (감사 개요)", "", chunks, "overview")
                stats["overview_chunks"] = len(chunks)
    else:
        all_chunks = []
        for title, sec_lines in split_sections(lines):
            for ch in chunk_paragraphs(to_paragraphs(sec_lines), opts.max_chars, opts.min_chars):
                ch["section"] = title
                all_chunks.append(ch)
        all_chunks = merge_small(all_chunks, opts.max_chars, opts.min_chars)
        parts = partition(all_chunks, opts.max_chunks_per_source) if all_chunks else []
        for k, part in enumerate(parts, 1):
            pid, ptitle = part_ident(f"AUD-{doc_id}-DOC", doc_title, k, len(parts))
            src = build_source(pid, ptitle, path, meta, common, "\n".join(c["text"] for c in part))
            for idx, ch in enumerate(part):
                sec = ch["section"]
                header = " > ".join(p for p in [ptitle, sec] if p)
                payload = {k2: [] for k2 in EMPTY_TAGS}
                if not opts.minimal_payload:
                    payload.update(base, kind="section", chunk_pages=[ch["page_start"], ch["page_end"]])
                    if sec:
                        payload["section"] = sec
                row = dict(src)
                row.update({"chunk_index": idx, "text": ch["text"],
                            "embedding_text": clip(f"{header}\n{ch['text']}", 32768),
                            "locator": make_locator(ch["page_start"], ch["page_end"], clip(sec, 40)),
                            "payload": payload})
                rows.append(row)
        stats["sources"] += len(parts)
        stats["split_sources"] += len(parts) > 1
    stats["chunks"] = len(rows)
    return {"rows": rows, "stats": stats}


# ---------------------------------------------------------------- 7. 검증·저장
def validate_rows(rows):
    """ingest_via_api.validate_request 로 원본별 요청을 검사한다. (errors, warns)"""
    import ingest_via_api as core
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["external_id"], []).append(r)
    errors, warns = [], []
    for ext, grp in groups.items():
        first = grp[0]
        src = {k: first[k] for k in first if k in ("source_type", "external_id", "title", "url", "file_path",
                                                    "publisher", "license", "content_sha256", "flagged")}
        chunks = [{k: v for k, v in r.items() if k in ("chunk_index", "text", "embedding_text", "locator",
                                                       "token_count", "payload")} for r in grp]
        e, w, _ = core.validate_request({"source": src, "chunks": chunks})
        errors += [f"{ext}: {m}" for m in e]
        warns += [f"{ext}: {m}" for m in w if "임베딩 서버가 한 번에" not in m]
    return errors, warns


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def say(msg=""):
    print(msg.encode("cp949", "replace").decode("cp949") if sys.stdout.encoding and
          sys.stdout.encoding.lower().startswith("cp949") else msg)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="감사 보고서 PDF 를 청크 JSONL 로 만든다(load_chunks.py 입력).")
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF 파일 또는 PDF 가 든 폴더(.txt 는 추출된 텍스트로 처리)")
    ap.add_argument("-o", "--out", type=Path, default=Path("audit_chunks.jsonl"), help="청크 JSONL 경로")
    ap.add_argument("--manifest", type=Path, help="파일별 doc_id·title·publisher·year·url·license 를 담은 JSON")
    ap.add_argument("--publisher", help="manifest 에 없을 때 쓸 기관명(100자 이하)")
    ap.add_argument("--license", help="manifest 에 없을 때 쓸 이용 조건(50자 이하)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help=f"청크 최대 글자 수(기본 {DEFAULT_MAX_CHARS})")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help=f"끝 조각을 앞에 붙이는 기준(기본 {DEFAULT_MIN_CHARS})")
    ap.add_argument("--max-chunks-per-source", type=int, default=DEFAULT_MAX_CHUNKS, metavar="N",
                    help=f"원본 하나의 청크 수 상한(기본 {DEFAULT_MAX_CHUNKS}). 넘으면 원본을 -P1, -P2 … 로 나눈다")
    ap.add_argument("--overview", choices=("skip", "include"), default="skip",
                    help="지적사항이 잡힌 문서의 앞부분(감사 개요)을 넣을지(기본 skip)")
    ap.add_argument("--use-headings", action="store_true",
                    help="지적사항 앞의 장 제목(Ⅰ. 설계단계 등)을 locator·임베딩 머리말·payload 에 쓴다(기본 끔)")
    ap.add_argument("--minimal-payload", action="store_true",
                    help="payload 에 분류 태그 빈 배열만 넣는다(서버가 추가 키를 거부할 때)")
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="문서마다 처음 N개 청크를 화면에 보여 준다")
    ap.add_argument("--no-validate", action="store_true", help="ingest_via_api 검증을 건너뛴다")
    ap.add_argument("--stats", type=Path, help="문서별 집계를 JSON 으로 저장한다")
    return ap.parse_args(argv)


def collect_inputs(inputs):
    files = []
    for p in inputs:
        if p.is_dir():
            files += sorted(x for x in p.iterdir() if x.suffix.lower() in (".pdf", ".txt"))
        else:
            files.append(p)
    return files


def run(args) -> int:
    if args.max_chunks_per_source < 1:
        say("[오류] --max-chunks-per-source 는 1 이상이어야 한다")
        return 2
    if args.max_chars < 200 or args.min_chars < 0 or args.min_chars >= args.max_chars:
        say("[오류] --max-chars 는 200 이상, --min-chars 는 0 이상이면서 --max-chars 보다 작아야 한다")
        return 2
    try:
        manifest = load_manifest(args.manifest)
        files = collect_inputs(args.inputs)
        if not files:
            raise InputError("처리할 PDF 가 없다")
        for opt in ("publisher", "license"):
            lim = 100 if opt == "publisher" else 50
            if getattr(args, opt) and len(getattr(args, opt)) > lim:
                raise InputError(f"--{opt} 는 {lim}자 이하여야 한다")
        common = {"publisher": args.publisher, "license": args.license}
        all_rows, all_stats, seen_ids = [], [], {}
        for f in files:
            res = process_document(f, manifest, common, args)
            doc_id = res["stats"]["doc_id"]
            if doc_id in seen_ids:
                raise InputError(f"doc_id {doc_id!r} 가 {seen_ids[doc_id]} 와 {f.name} 에서 겹친다. manifest 로 구분한다")
            seen_ids[doc_id] = f.name
            all_rows += res["rows"]
            all_stats.append(res["stats"])
    except InputError as e:
        say(f"[오류] {e}")
        return 2

    if not all_rows:
        say("[오류] 만들어진 청크가 없다")
        return 2
    if not args.no_validate:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        errors, warns = validate_rows(all_rows)
        for w in warns[:20]:
            say(f"[경고] {w}")
        if errors:
            for e in errors[:30]:
                say(f"[오류] {e}")
            say(f"[오류] 검증 오류 {len(errors)}건. 파일을 쓰지 않았다")
            return 2
    write_jsonl(args.out, all_rows)
    if args.stats:
        args.stats.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    say(f"{'문서':<8}{'쪽':>5}  {'방식':<9}{'지적':>5}{'청크':>6}  파일")
    for s in all_stats:
        say(f"{s['doc_id']:<8}{s['pages']:>5}  {s['strategy']:<9}{s['findings']:>5}{s['chunks']:>6}  {s['file']}")
    ids = {r["external_id"] for r in all_rows}
    say(f"합계: 문서 {len(all_stats)}개, 원본 {len(ids)}개, 청크 {len(all_rows)}개 -> {args.out}")
    split = sum(s["split_sources"] for s in all_stats)
    if split:
        say(f"청크가 {args.max_chunks_per_source}개를 넘어 나눈 원본: {split}개(이름 끝에 -P1, -P2 …)")
    if args.preview:
        shown = {}
        for r in all_rows:
            did = r["payload"].get("doc_id", "?")
            if shown.get(did, 0) < args.preview:
                shown[did] = shown.get(did, 0) + 1
                say(f"--- {r['external_id']} #{r['chunk_index']} [{r['locator']}] {r['title'][:50]}")
                say(r["text"][:300].replace("\n", " / "))
    say("다음: python scripts/load_chunks.py " + str(args.out) + "  (검증만. 보내려면 --send)")
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
