"""공통 작은 도구. 레코드는 dict로 다룬다(JSON 중간 파일 그대로)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = 1


def nonspace(text: str | None) -> int:
    """공백이 아닌 글자 수. 글자 보존 장부의 단위."""
    return len("".join((text or "").split()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """키 정렬 JSON의 sha256. 같은 내용이면 항상 같은 값."""
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def one_line(text: str | None) -> str:
    """이름·분류처럼 한 줄 값: 연속 공백을 한 칸으로."""
    return " ".join((text or "").split())
