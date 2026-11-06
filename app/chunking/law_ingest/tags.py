"""계층 경로(편 > 장 > 절)로 hazard / work_type 태그를 붙이는 규칙 (초안, 기본 꺼짐).

코드 값은 A님 문서 4-2장의 예시다(work_type: GENERAL·HOT·CONFINED·HEIGHT·ELECTRICAL·LIFTING,
hazard: MECHANICAL·ELECTRICAL·CHEMICAL·ENVIRONMENTAL·HUMAN). 업무 DB `tag_codes`의 실제 값과
맞는지는 `GET /api/v1/tag-codes` 응답으로 확인해야 한다(확인 필요).

잘못 붙이면 5축 필터 검색에서 그 청크가 빠지므로, 불확실한 항목은 비워 둔다.
켜기: config.TAGS_ENABLED = True 또는 --tags.
"""

from __future__ import annotations

import re

# (경로에 이 정규식이 있으면, work_type 목록, hazard 목록). 위에서부터 모두 적용해 합친다.
TAG_RULES: list[tuple[str, list[str], list[str]]] = [
    (r"제2절 화기 등의 관리", ["HOT"], []),
    (r"제10장 밀폐공간", ["CONFINED"], []),
    (r"제3장 전기로 인한 위험 방지", ["ELECTRICAL"], ["ELECTRICAL"]),
    (r"제9절 양중기", ["LIFTING"], ["MECHANICAL"]),
    (r"제1절 추락에 의한 위험 방지", ["HEIGHT"], []),
    (r"제7장 비계", ["HEIGHT"], []),
    (r"제1장 기계.{1,3}기구 및 그 밖의 설비에 의한 위험예방", [], ["MECHANICAL"]),
    (r"제1장 관리대상 유해물질", [], ["CHEMICAL"]),
    (r"제2장 허가대상 유해물질", [], ["CHEMICAL"]),
    (r"제3장 금지유해물질", [], ["CHEMICAL"]),
    (r"제12장 근골격계부담작업", [], ["HUMAN"]),
]

# 조 단위 예외(절 경계와 주제가 어긋나는 조). 키는 article.key. 지금은 비어 있음.
ARTICLE_OVERRIDES: dict[str, tuple[list[str], list[str]]] = {}


def tags_for(path_labels: list[str], article_key: str) -> tuple[list[str], list[str]]:
    """(work_type, hazard)."""
    if article_key in ARTICLE_OVERRIDES:
        wt, hz = ARTICLE_OVERRIDES[article_key]
        return list(wt), list(hz)
    path = " > ".join(path_labels)
    work_type: list[str] = []
    hazard: list[str] = []
    for pattern, wts, hzs in TAG_RULES:
        if re.search(pattern, path):
            work_type.extend(w for w in wts if w not in work_type)
            hazard.extend(h for h in hzs if h not in hazard)
    return work_type, hazard
