"""프롬프트 인젝션 의심 문장 검사. 소스 단위로 판정한다(하나라도 걸리면 소스 전체 flagged).

공공 데이터라 거의 걸리지 않는다. 걸린 소스는 flagged=true 로 올라가 검색에서 빠진다.
"""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)", re.I)),
    ("disregard_instructions", re.compile(r"disregard\s+.{0,30}instruction", re.I)),
    ("system_prompt", re.compile(r"system\s*prompt", re.I)),
    ("ko_ignore_previous", re.compile(r"이전\s*(지시|명령|지침|프롬프트).{0,12}(무시|잊)")),
    ("ko_system_prompt", re.compile(r"시스템\s*프롬프트")),
    ("script_tag", re.compile(r"<\s*script", re.I)),
)


def scan(text: str) -> list[str]:
    return [name for name, rx in _PATTERNS if rx.search(text)]
