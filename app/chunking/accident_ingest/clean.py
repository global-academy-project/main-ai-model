"""S3 정리: 서술 필드의 마크업·공백을 정리하고 글자 보존을 장부로 남긴다.

하는 일: HTML 태그·엔티티 제거, \\r·nbsp·제어문자 정리, 연속 공백·빈 줄 정리, 앞뒤 공백 제거.
하지 않는 일: 문장 수정, 맞춤법, 단위 통일, 익명화 표기(○○/OO) 변경, ▶·『』 같은 구조 기호 제거,
월의 0 채움 변경.

필드마다 `ledger`에 공백 아닌 글자 수를 적는다.
    raw = kept + markup + entities + control
검증(validate)이 raw 값에서 이 식을 다시 계산한다.
"""

from __future__ import annotations

import html
import re
from typing import Any

from . import config
from .models import nonspace, one_line

NARRATIVE_FIELDS = tuple(field for _, field, _ in config.SECTIONS)
# 분류·이름처럼 한 줄 값으로만 쓰는 필드(제목·머리말·태그에 쓰인다)
META_FIELDS = ("cateSeNm", "sifLclsfNm", "sifMclsfNm", "sifSclsfNm", "disasterType", "orgtNm", "situation")
KNOWN_FIELDS = frozenset(NARRATIVE_FIELDS) | frozenset(META_FIELDS)

_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1b\x7f​‌‍⁠﻿]")
_SPACES_RE = re.compile(r"[ \t 　]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def clean_text(raw: str | None) -> tuple[str, dict[str, int]]:
    s = raw or ""
    ledger = {"raw": nonspace(s), "kept": 0, "markup": 0, "entities": 0, "control": 0}

    before = nonspace(s)
    s = _TAG_RE.sub(" ", s)
    ledger["markup"] = before - nonspace(s)

    before = nonspace(s)
    s = _ENTITY_RE.sub(lambda m: html.unescape(m.group(0)), s)
    ledger["entities"] = before - nonspace(s)

    before = nonspace(s)
    s = _CTRL_RE.sub("", s)
    ledger["control"] = before - nonspace(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES_RE.sub(" ", ln).strip() for ln in s.split("\n")]
    s = _BLANKS_RE.sub("\n\n", "\n".join(lines)).strip()
    ledger["kept"] = nonspace(s)
    return s, ledger


def is_empty(text: str) -> bool:
    return text.strip().lower() in {v.lower() for v in config.EMPTY_VALUES}


def clean_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """map 단계의 문서 하나 → 정리된 문서. 원본 값(`raw`)과 정리 결과(`clean`)를 함께 둔다."""
    raw = {f: doc.get(f) for f in (*NARRATIVE_FIELDS, *META_FIELDS)}
    clean: dict[str, str] = {}
    ledger: dict[str, dict[str, int]] = {}
    empty: list[str] = []
    for f in NARRATIVE_FIELDS:
        text, led = clean_text(doc.get(f))
        if is_empty(text):
            empty.append(f)
            text = ""
        clean[f] = text
        ledger[f] = led
    for f in META_FIELDS:
        clean[f] = one_line(doc.get(f))
    unknown = sorted(k for k in doc if not k.startswith("_") and k not in KNOWN_FIELDS and k not in _MAP_KEYS)
    out = {k: v for k, v in doc.items() if k not in KNOWN_FIELDS}
    out.update({"raw": raw, "clean": clean, "ledger": ledger, "empty_sections": empty, "unknown_fields": unknown})
    return out


# map 단계가 문서에 더하는 키(알 수 없는 필드로 세지 않는다)
_MAP_KEYS = frozenset({"external_id", "id_base", "id_suffix", "tier", "selected", "content_sha256", "_src"})
