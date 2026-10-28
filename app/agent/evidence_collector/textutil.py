"""문자열·목록 정리 도우미와 LLM 출력 JSON 추출."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json_object(raw: str) -> dict | None:
    """모델 출력에서 JSON 객체를 꺼낸다. <think> 블록, ```json 펜스, 앞뒤 잡담을 견딘다."""
    text = _THINK_RE.sub("", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def str_list(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def merge(*lists: Sequence[str]) -> list[str]:
    """순서를 지키며 중복을 뺀다."""
    out: list[str] = []
    for items in lists:
        for item in items:
            if item not in out:
                out.append(item)
    return out


def one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]
